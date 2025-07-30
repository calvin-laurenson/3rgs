import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import sys
from pathlib import Path
import torch.nn.functional as F
import copy
model = VGGT()
model.load_state_dict(torch.load("./vggt_1B_commercial.pt", map_location="cpu"))
model.to("cuda")

input_paths = [str(v) for v in Path(sys.argv[1]).iterdir()]
images, original_coords = load_and_preprocess_images_square(input_paths, 1024)
images = images.to("cuda")
assert len(images.shape) == 4
assert images.shape[1] == 3

# hard-coded to use 518 for VGGT
vggt_images = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        vggt_images = vggt_images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(vggt_images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, vggt_images.shape[-2:])
    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, vggt_images, ps_idx)

extrinsic = extrinsic.squeeze(0).cpu().numpy()
intrinsic = intrinsic.squeeze(0).cpu().numpy()
depth_map = depth_map.squeeze(0).cpu().numpy()
depth_conf = depth_conf.squeeze(0).cpu().numpy()

points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

conf_thres_value = 5.0
max_points_for_colmap = 100000  # randomly sample 3D points
shared_camera = False  # in the feedforward manner, we do not support shared camera
camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

image_size = np.array([518, 518])
num_frames, height, width, _ = points_3d.shape
print(images.shape)
points_rgb = F.interpolate(
    images, size=(518, 518), mode="bilinear", align_corners=False
)
points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
points_rgb = points_rgb.transpose(0, 2, 3, 1)

# (S, H, W, 3), with x, y coordinates and frame indices
points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

conf_mask = depth_conf >= conf_thres_value
# at most writing 100000 3d points to colmap reconstruction object
conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

points_3d = points_3d[conf_mask]
points_xyf = points_xyf[conf_mask]
points_rgb = points_rgb[conf_mask]

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = (image_paths[pyimageid - 1]).split("/")[-1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        input_paths,
        original_coords.cpu().numpy(),
        img_size=518,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

sparse_reconstruction_dir = Path(sys.argv[2]) / "sparse"
sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
reconstruction.write(sparse_reconstruction_dir)