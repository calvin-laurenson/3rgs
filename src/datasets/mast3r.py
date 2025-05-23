import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from plyfile import PlyData, PlyElement
from PIL import Image
import tqdm
import torch.nn.functional as F
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
import evo.core.geometry as geometry

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """DUSR3R parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        image_fn_ext = 'jpg',
    ):
        #colmap_image_dir = os.path.join(data_dir, "images"+f"_{factor}")
        colmap_image_dir = os.path.join(data_dir, "images")
        colmap_images = sorted(os.listdir(colmap_image_dir))
        image_fn_ext = colmap_images[0].split('.')[-1]

        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        dust_dir = os.path.join(data_dir, "mast3r/0/")
        if not os.path.exists(dust_dir):
            dust_dir = os.path.join(data_dir, "mast3r")
        assert os.path.exists(
            dust_dir
        ), f"mast3r directory {dust_dir} does not exist."
        self.dust_dir = dust_dir

        with open(os.path.join(data_dir, 'images_train.txt'), 'r') as file:
            self.train_split = [line.strip() for line in file]
        with open(os.path.join(data_dir, 'images_test.txt'), 'r') as file:
            self.test_split = [line.strip() for line in file]

        intrinsics_train = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
        camtoworlds_train = np.load(os.path.join(dust_dir, 'camera_poses.npy'))

        camtoworlds_train_gt = np.load(os.path.join(data_dir, 'pose_gt_train.npy'))
        camtoworlds_test_gt = np.load(os.path.join(data_dir, 'pose_gt_test.npy'))

        use_test_camera = True
        if use_test_camera:
            transform = align_pose(camtoworlds_train_gt, camtoworlds_train)
            camtoworlds_test = np.einsum('ij,bjk->bik', transform, camtoworlds_test_gt)  # Apply transformation to ground truth poses
            intrinsics_test = np.tile(intrinsics_train.mean(axis=0), (len(self.test_split), 1, 1))

            intrinsics = np.concatenate([intrinsics_train, intrinsics_test], axis=0)
            camtoworlds = np.concatenate([camtoworlds_train, camtoworlds_test], axis=0)
            camtoworlds_gt = np.concatenate([camtoworlds_train_gt, camtoworlds_test_gt], axis=0)

            filelist = [f"{img}.{image_fn_ext}" for img in self.train_split + self.test_split]
        else:
            # test with gt camera pose
            self.test_split = list(self.train_split)
            intrinsics = intrinsics_train
            camtoworlds = camtoworlds_train
            camtoworlds_gt = camtoworlds_train_gt

            filelist = [f"{img}.{image_fn_ext}" for img in self.train_split]

        inds = np.argsort(filelist)
        filelist = [filelist[i] for i in inds]
        intrinsics = intrinsics[inds]
        camtoworlds = camtoworlds[inds]
        camtoworlds_gt = camtoworlds_gt[inds]
        self.intrinsics = intrinsics

        ply_data = PlyData.read(os.path.join(dust_dir, 'pointcloud.ply'))
        point_cloud = ply_data['vertex']
        points = np.stack([point_cloud['x'], point_cloud['y'], point_cloud['z']], axis=-1)

        self.image_names = filelist  # List[str], (num_images,)
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        
        # Create resized image directory if it doesn't exist
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"Created directory {image_dir}")
            
            # Resize and save all images from colmap directory
            colmap_images = sorted(os.listdir(colmap_image_dir))
            for img_name in tqdm.tqdm(colmap_images, desc="Resizing images"):
                img_path = os.path.join(colmap_image_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                new_h, new_w = h // factor, w // factor
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                cv2.imwrite(os.path.join(image_dir, img_name), img_resized)


        self.image_paths = [os.path.join(data_dir, 'images' + image_dir_suffix, file) for file in filelist]  # List[str], (num_images,)

        # handle factor
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        factor_intr = 512 / actual_width
        self.intrinsics[:,:2,:] /= factor_intr
        self.image_size = (actual_width, actual_height)
        self.factor_intr = factor_intr

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camtoworlds_gt = camtoworlds_gt # np.ndarray, (num_images, 4, 4)
        self.camera_ids = [i + 1 for i in range(camtoworlds.shape[0])]  # List[int], (num_images,)
        self.Ks_dict = {cam_id: self.intrinsics[i] for i, cam_id in enumerate(self.camera_ids)}  # Dict of camera_id -> K

        # use gt intrinsics
        use_gt_intrinsics = True
        if use_gt_intrinsics:
            colmap_dir = os.path.join(data_dir, "sparse/0/")
            print(colmap_dir)
            if not os.path.exists(colmap_dir):
                colmap_dir = os.path.join(data_dir, "sparse")
            assert os.path.exists(
                colmap_dir
            ), f"COLMAP directory {colmap_dir} does not exist."
            manager = SceneManager(colmap_dir)
            manager.load_cameras()
            camera_id = 1
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            self.Ks_dict = {cam_id: K 
                for i, cam_id in enumerate(self.camera_ids)
            }            

        self.intrinsics[:,:,:] = self.Ks_dict[1]
        self.params_dict = {cam_id: [] for cam_id in self.camera_ids}  # Dict of camera_id -> params
        self.imsize_dict = {camera_id: Image.open(image_path).size for camera_id, image_path in
                            zip(self.camera_ids, self.image_paths)}  # Dict of camera_id -> (width, height)
        self.mask_dict = {cam_id: None for cam_id in self.camera_ids}  # Dict of camera_id -> mask
        self.points = points
        self.points_err = np.zeros_like(self.points)  # np.ndarray, (num_points,)
        self.points_rgb = np.stack([point_cloud['red'], point_cloud['green'], point_cloud['blue']], axis=-1)  # np.ndarray, (num_points, 3)


        num_sampled = 150000
        num_points = self.points.shape[0]
        pt_indices = np.random.choice(num_points, size=num_sampled, replace=False)
        self.points_sampled = self.points[pt_indices]
        self.points_rgb_sampled = self.points_rgb[pt_indices]
        self.points_err_sampled = self.points_err[pt_indices]  # 假设误差为0

        # Add random noise to points
        use_noise_init = False if 'scan' in data_dir else True # DTU scene use pcls to init
        if use_noise_init:
            self.points = self.points_sampled
            self.points_err = self.points_err_sampled
            self.points_rgb = self.points_rgb_sampled
            self.points = np.random.normal(0, 1.5, self.points.shape) 
        else:
            self.points = self.points_sampled
            self.points_err = self.points_err_sampled
            self.points_rgb = self.points_rgb_sampled

        self.point_indices = {} # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = np.zeros((4, 4))  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

def align_pose(pose_a, pose_b):
    # Calculate alignment parameters using umeyama
    r, t, c = geometry.umeyama_alignment(pose_a[:,:3,3].T, pose_b[:,:3,3].T, with_scale=True)
            
    # Create 4x4 transformation matrix
    device = pose_a.device if torch.is_tensor(pose_a) else torch.device('cpu')
    transform = torch.eye(4, device=device)
    transform[:3,:3] = c * torch.from_numpy(r).to(device).float()  # Apply rotation and scale
    transform[:3,3] = torch.from_numpy(t).to(device).float()  # Add translation

    return transform


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        verbose: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        name_list = [os.path.splitext(name)[0] for name in self.parser.image_names]
        if split == "train":
            self.indices = [name_list.index(name) for name in self.parser.train_split]
        else:
            self.indices = [name_list.index(name) for name in self.parser.test_split]
        self.intrinsics = self.parser.intrinsics[self.indices]
        self.camtoworlds = self.parser.camtoworlds[self.indices]
        self.camtoworlds_gt = self.parser.camtoworlds_gt[self.indices]

        # Calculate scale between estimated and ground truth camera positions
        est_positions = self.camtoworlds[:, :3, 3]  # [N, 3]
        # Center both point sets
        est_center = est_positions.mean(axis=0, keepdims=True)  # [1, 3]
        est_centered = est_positions - est_center  # [N, 3]
        # Calculate scale as ratio of RMS distances from center
        est_scale = np.sqrt(np.mean(np.sum(est_centered**2, axis=1)))
        self.cam_scale = est_scale
        self.image_size = self.parser.image_size

        if verbose:
            print(split, self.indices)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        camtoworlds_gt = self.parser.camtoworlds_gt[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "camtoworld_gt": torch.from_numpy(camtoworlds_gt).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data

class CorrespondenceDataset():
    """Dataset class for loading correspondence data between image pairs."""
    
    def __init__(self, parser: Parser, split: str = "train", patch_size: Optional[int] = None, load_depths: bool = False):
        """Initialize the dataset.
        
        Args:
            parser (Parser): Parser object containing dataset info
            split (str): Dataset split ("train" or "test")
            patch_size (Optional[int]): Size of random patches to extract
            load_depths (bool): Whether to load depth information
        """
        # Initialize base dataset
        self.dataset = Dataset(parser, split, patch_size, load_depths)
        
        data_dir = parser.dust_dir
        
        # Load correspondence tensors from numpy arrays
        self.corr_i = torch.from_numpy(np.load(os.path.join(data_dir, 'corr_i.npy')))
        self.corr_j = torch.from_numpy(np.load(os.path.join(data_dir, 'corr_j.npy')))
        self.corr_batch_idx = torch.from_numpy(np.load(os.path.join(data_dir, 'corr_batch_idx.npy')))
        self.corr_mask = torch.from_numpy(np.load(os.path.join(data_dir, 'corr_mask.npy')))
        self.corr_weight = torch.from_numpy(np.load(os.path.join(data_dir, 'corr_weight.npy')))
        self.ei = torch.from_numpy(np.load(os.path.join(data_dir, 'ei.npy')))
        self.ej = torch.from_numpy(np.load(os.path.join(data_dir, 'ej.npy')))
        self.depthmaps = torch.from_numpy(np.load(os.path.join(data_dir, 'depthmaps.npy')))
        # Resize depthmaps to match dataset image size
        self.depthmaps = F.interpolate(
            self.depthmaps.unsqueeze(1), # Add channel dimension
            size=self.dataset.image_size[::-1],
            mode='bilinear',
            align_corners=False
        ).squeeze(1) # Remove channel dimension

        # uniform downsample
        pairs_step = 1
        cores_step = 20
        self.ei = self.ei[::pairs_step]
        self.ej = self.ej[::pairs_step]
        self.corr_i = self.corr_i[::pairs_step,::cores_step]
        self.corr_j = self.corr_j[::pairs_step,::cores_step]
        self.corr_weight = self.corr_weight[::pairs_step,::cores_step]
        self.corr_mask = self.corr_mask[::pairs_step,::cores_step]
        self.corr_batch_idx = self.corr_batch_idx[:len(self.ei),::cores_step]
        
        # Verify all tensors have same first dimension
        self.length = len(self.corr_i)
        assert len(self.corr_j) == self.length
        assert len(self.corr_batch_idx) == self.length
        assert len(self.corr_mask) == self.length
        assert len(self.corr_weight) == self.length
        assert len(self.ei) == self.length
        assert len(self.ej) == self.length

        # handle factor
        factor_intr = self.dataset.parser.factor_intr
        x_i = torch.floor(self.corr_i % 512 / factor_intr).long()
        y_i = torch.floor(self.corr_i // 512 / factor_intr).long()
        x_j = torch.floor(self.corr_j % 512 / factor_intr).long()
        y_j = torch.floor(self.corr_j // 512 / factor_intr).long()
        # Clip coordinates to image boundaries
        x_i = torch.clamp(x_i, 0, self.dataset.image_size[0] - 1)
        y_i = torch.clamp(y_i, 0, self.dataset.image_size[1] - 1)
        x_j = torch.clamp(x_j, 0, self.dataset.image_size[0] - 1)
        y_j = torch.clamp(y_j, 0, self.dataset.image_size[1] - 1)

        #print(x_i.max(), y_i.max(), x_j.max(), y_j.max())
        #print(self.dataset.image_size)
        self.corr_i = (y_i * self.dataset.image_size[0] + x_i).long()
        self.corr_j = (y_j * self.dataset.image_size[0] + x_j).long()

        # find correspondence points in image space
        self.corr_points_i = xy_grid(self.dataset.image_size[0], self.dataset.image_size[1], device=self.corr_i.device).reshape(-1, 2)
        self.corr_points_j = xy_grid(self.dataset.image_size[0], self.dataset.image_size[1], device=self.corr_j.device).reshape(-1, 2)
        self.corr_points_i = self.corr_points_i[self.corr_i].float()
        self.corr_points_j = self.corr_points_j[self.corr_j].float()

        ## Visualize correspondence points
        #match_dir = 'match_viz'
        #os.makedirs(match_dir, exist_ok=True)
        ## Create sample visualization for first few pairs
        #for i in range(min(5, len(self.ei))):
            ## Get images for this pair
            #image_i = self.dataset[self.ei[i]]['image'].numpy() /255.
            #image_j = self.dataset[self.ej[i]]['image'].numpy() /255.
            
            ## Get correspondence points for this pair
            #points_i = self.corr_points_i[i].detach().cpu().numpy()
            #points_j = self.corr_points_j[i].detach().cpu().numpy()
            #mask = self.corr_mask[i].detach().cpu().numpy()
            
            ## Only keep valid correspondences
            #points_i = points_i[mask > 0]
            #points_j = points_j[mask > 0]
            
            ## Sample subset of points if too many
            #n_viz = min(20, len(points_i))
            #if len(points_i) > n_viz:
                #idx = np.round(np.linspace(0, len(points_i)-1, n_viz)).astype(int)
                #points_i = points_i[idx]
                #points_j = points_j[idx]

            ## Create side-by-side visualization
            #H0, W0, H1, W1 = *image_i.shape[:2], *image_j.shape[:2]
            #img0 = np.pad(image_i, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
            #img1 = np.pad(image_j, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
            #img = np.concatenate((img0, img1), axis=1)
            
            #import matplotlib.pyplot as plt  # Add this import
            #plt.figure(figsize=(12,6))
            #plt.imshow(img)
            #cmap = plt.get_cmap('jet')
            #for j in range(n_viz):
                #x0, y0 = points_i[j]
                #x1, y1 = points_j[j]
                #plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(j / (n_viz - 1)), scalex=False, scaley=False)
            #plt.title(f'Correspondence visualization for pair {i}')
            #plt.savefig(os.path.join(match_dir, f'matches_pair_{i}.png'))
            #plt.close()

        self.camtoworlds_all = torch.from_numpy(self.dataset.camtoworlds).float()
        self.camtoworlds_gt_all = torch.from_numpy(self.dataset.camtoworlds_gt).float()
        self.epipolar_data = {
            'intrinsics_i_44_all': intrinsic_to_44(torch.from_numpy(self.dataset.intrinsics[self.ei]).float()),
            'intrinsics_j_44_all': intrinsic_to_44(torch.from_numpy(self.dataset.intrinsics[self.ej]).float()),
            'camtoworlds_i_all': torch.from_numpy(self.dataset.camtoworlds[self.ei]).float(),
            'camtoworlds_j_all': torch.from_numpy(self.dataset.camtoworlds[self.ej]).float(),
            'ei_all': self.ei,
            'ej_all': self.ej,
            'corr_mask_all': self.corr_mask,
            'corr_weight_all': self.corr_weight,
            'corr_points_i_all': self.corr_points_i,
            'corr_points_j_all': self.corr_points_j,
        }

    def __len__(self):
        """Return the number of image pairs."""
        return self.length

    def __getitem__(self, idx):
        """Get correspondence data for a specific image pair.
        
        Args:
            idx (int): Index of the image pair
            
        Returns:
            dict: Dictionary containing base dataset and correspondence data
        """
        # Get base dataset items for both images

        #data_i = self.dataset[idx]
        #data_j = self.dataset[idx+1]

        data_i = self.dataset[self.ei[idx]]
        data_j = self.dataset[self.ej[idx]]

        
        # Concatenate image, camtoworld and K data
        data = {
            'image': torch.stack([data_i['image'], data_j['image']]),
            'camtoworld': torch.stack([data_i['camtoworld'], data_j['camtoworld']]),
            'camtoworld_gt': torch.stack([data_i['camtoworld_gt'], data_j['camtoworld_gt']]),
            'K': torch.stack([data_i['K'], data_j['K']]),
            "image_id": torch.stack([self.ei[idx], self.ej[idx]]),
            'idx': idx
        }
        
        # Add correspondence data
        corr_data = {
            'corr_i': self.corr_i[idx].reshape(1, -1),  
            'corr_j': self.corr_j[idx].reshape(1, -1),  
            'corr_batch_idx': self.corr_batch_idx[0].reshape(1, -1),  # batch_size = 1
            'corr_mask': self.corr_mask[idx].reshape(1, -1),  
            'corr_weight': self.corr_weight[idx].reshape(1, -1),  
            'ei': self.ei[idx],  
            'ej': self.ej[idx],
            'depthmaps': torch.stack([self.depthmaps[self.ei[idx]], self.depthmaps[self.ej[idx]]])
        }
        
        # Merge dictionaries
        data.update(corr_data)

        return data

def intrinsic_to_44(K):
    """Convert Bx3x3 intrinsic matrix to Bx4x4 homogeneous projection matrix.
            
    Args:
        K (torch.Tensor): Bx3x3 intrinsic matrix
                
    Returns:
        torch.Tensor: Bx4x4 homogeneous projection matrix
    """
    device = K.device
    batch_size = K.shape[0]
    K_44 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    K_44[:,:3,:3] = K
    return K_44
def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
