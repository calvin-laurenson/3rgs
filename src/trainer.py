import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import kornia
#from datasets.colmap import Dataset, Parser
from datasets.mast3r import Dataset, Parser, CorrespondenceDataset, align_pose
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from utils.cam_utils import AppearanceOptModule, knn, rgb_to_sh, set_random_seed

from utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from utils.eval_utils import eval_ate



@dataclass
class Config:
    # use epipolar loss
    use_corres_epipolar_loss: bool = True 
    epi_loss_weight: float = 2.0 

    # Enable camera optimization.
    pose_opt: bool = True
    pose_opt_type: Literal["sfm", "mlp"] = "mlp" 
    # Learning rate for camera optimization
    pose_opt_lr: float = None 
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 0
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0

    # Enable camera intrinsics optimization
    intrinsics_opt: bool = False
    # Learning rate for intrinsics optimization
    focal_opt_lr: float = 0.0001
    pp_opt_lr: float = 0
    # Regularization for intrinsics optimization as weight decay  
    focal_opt_reg: float = 0 #1e-5
    pp_opt_reg: float = 0 
    # Add noise to camera intrinsics. This is only to test the intrinsics optimization.
    intrinsics_noise: float = 0.0

    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0



    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            verbose=True
        )
        self.trainvalset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val", verbose=True)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            if cfg.pose_opt_type == "sfm":
                from utils.cam_utils import CameraOptModule
            elif cfg.pose_opt_type == "mlp":
                from utils.cam_utils import CameraOptModuleMLP as CameraOptModule
            self.pose_adjust = CameraOptModule(len(self.trainset), trainset=self.trainset).to(self.device)
            self.pose_adjust.zero_init()

            if cfg.pose_opt_type == "mlp":
                cfg.pose_opt_lr = 15e-5 
            elif cfg.pose_opt_type == "sfm":
                cfg.pose_opt_lr = 1e-5

            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        self.intrinsics_optimizers = []
        if cfg.intrinsics_opt:
            K = torch.tensor(self.trainset.intrinsics).float().mean(dim=0)
            focal = (K[0,0] + K[1,1]) / 2
            pp = K[:2,2]
            # Convert to leaf tensors before creating parameters
            focal = torch.tensor(focal.item(), device=self.device)
            pp = torch.tensor(pp.tolist(), device=self.device) 
            self.imsize = torch.tensor(self.trainset.parser.image_size, device=self.device).float()
            self.focal_opt = torch.nn.Parameter(focal.log())
            self.pp_opt = torch.nn.Parameter(pp/self.imsize)
            self.intrinsics_optimizers = [
                torch.optim.Adam(
                    [self.focal_opt],
                    lr=cfg.focal_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.focal_opt_reg,
                ),
                torch.optim.Adam(
                    [self.pp_opt],
                    lr=cfg.pp_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pp_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
    def optim_camtoworlds(self, camtoworlds, Ks, width, height, sh_degree, near_plane, far_plane, masks, pixels, image_id, show_progress=False):
        with torch.enable_grad():
            iters = 100 #* 2
            pose_opt_lr = 8e-4
            from utils.cam_utils import CameraOptModule
            pose_adjust = CameraOptModule(1).to(self.device)
            pose_adjust.zero_init()
            pose_optimizer = torch.optim.Adam(
                    pose_adjust.parameters(),
                    lr=pose_opt_lr,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        pose_optimizer, gamma=0.01 ** (1.0 / iters)
            )
        
            # Use tqdm only if show_progress is True
            iterator = tqdm.tqdm(range(iters), desc="Optimizing camera pose") if show_progress else range(iters)
            for i in iterator:
                camtoworlds_optim = pose_adjust(camtoworlds, torch.tensor([0],device=self.device))
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds_optim,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    masks=masks,
                    optim_3dgs=True
                )

                #loss = F.l1_loss(renders, pixels)
                # gradient loss
                loss = compute_gradient_loss(pixels, renders)

                loss.backward()
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
                # Update progress bar description only if show_progress is True
                if show_progress:
                    iterator.set_description(f"Optimizing camera pose (loss={loss.item():.6f})")

            # set gradients to none
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.intrinsics_optimizers:
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.zero_grad(set_to_none=True)
        return camtoworlds_optim.detach()


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        optim_3dgs: bool = True,  # Renamed parameter to control 3D Gaussian optimization
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Get splat attributes and optionally detach them to prevent gradient flow
        means = self.splats["means"] if optim_3dgs else self.splats["means"].detach()
        quats = self.splats["quats"] if optim_3dgs else self.splats["quats"].detach()
        scales = torch.exp(self.splats["scales"] if optim_3dgs else self.splats["scales"].detach())
        opacities = torch.sigmoid(self.splats["opacities"] if optim_3dgs else self.splats["opacities"].detach())

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            features = self.splats["features"] if optim_3dgs else self.splats["features"].detach()
            colors = self.app_module(
                features=features,
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + (self.splats["colors"] if optim_3dgs else self.splats["colors"].detach())
            colors = torch.sigmoid(colors)
        else:
            sh0 = self.splats["sh0"] if optim_3dgs else self.splats["sh0"].detach()
            shN = self.splats["shN"] if optim_3dgs else self.splats["shN"].detach()
            colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )


        # trainloader
        trainloader = torch.utils.data.DataLoader(
        self.trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        )
        if self.cfg.use_corres_epipolar_loss:
            trainset_corres = CorrespondenceDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.intrinsics_i_44_all = trainset_corres.epipolar_data['intrinsics_i_44_all'].to(device)
            self.intrinsics_j_44_all = trainset_corres.epipolar_data['intrinsics_j_44_all'].to(device)
            self.camtoworlds_i_all = trainset_corres.epipolar_data['camtoworlds_i_all'].to(device)
            self.camtoworlds_j_all = trainset_corres.epipolar_data['camtoworlds_j_all'].to(device)
            self.ei_all = trainset_corres.epipolar_data['ei_all'].to(device)
            self.ej_all = trainset_corres.epipolar_data['ej_all'].to(device)
            self.corr_mask_all = trainset_corres.epipolar_data['corr_mask_all'].to(device)
            self.corr_weight_all = trainset_corres.epipolar_data['corr_weight_all'].to(device)
            self.corr_points_i_all = trainset_corres.epipolar_data['corr_points_i_all'].to(device)
            self.corr_points_j_all = trainset_corres.epipolar_data['corr_points_j_all'].to(device)

        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        data =None
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            camtoworlds_epi = camtoworlds
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # intrinsics optimization
            if cfg.intrinsics_opt:
                Ks = torch.eye(3, dtype=torch.float32, device=device)[None].expand(camtoworlds.shape[0], 3, 3).clone()
                Ks[:, 0, 0] = Ks[:, 1, 1] = self.focal_opt.exp()
                Ks[:, 0:2, 2] = self.pp_opt * self.imsize
                Ks = Ks#.detach().clone() #####!!!!!!! careful
                if step % 50 == 0:
                    print('current intrinsics:')
                    print(Ks)
            if step == 0:
                print('current intrinsics:')
                print(Ks)
            
            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )


            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
            lcorres_2d = torch.tensor(0.0, device=self.device)

            # Epipolar loss
            lepipolar = torch.tensor(0.0, device=self.device)
            if cfg.use_corres_epipolar_loss and (step >0 and step < 3000): #and step < 3499 or step > 7400 and step < 10400):
                camtoworlds_i_all = self.camtoworlds_i_all
                camtoworlds_j_all = self.camtoworlds_j_all
                if cfg.pose_noise:
                    camtoworlds_i_all = self.pose_perturb(camtoworlds_i_all, self.ei_all)
                    camtoworlds_j_all = self.pose_perturb(camtoworlds_j_all, self.ej_all)
                if cfg.pose_opt:
                    camtoworlds_i_all = self.pose_adjust(camtoworlds_i_all, self.ei_all)
                    camtoworlds_j_all = self.pose_adjust(camtoworlds_j_all, self.ej_all)
                im_poses_i_all = torch.linalg.inv(camtoworlds_i_all)
                im_poses_j_all = torch.linalg.inv(camtoworlds_j_all)

                if cfg.intrinsics_opt:
                    Ks = torch.eye(4, dtype=torch.float32, device=self.device)[None].expand(camtoworlds_i_all.shape[0], 4, 4).clone()
                    Ks[:, 0, 0] = Ks[:, 1, 1] = self.focal_opt.exp()
                    Ks[:, 0:2, 2] = self.pp_opt * self.imsize
                    intrinsics_i_44_all = Ks
                    intrinsics_j_44_all = Ks
                else:
                    intrinsics_i_44_all = self.intrinsics_i_44_all
                    intrinsics_j_44_all = self.intrinsics_j_44_all

                P_i = intrinsics_i_44_all @ im_poses_i_all
                P_j = intrinsics_j_44_all @ im_poses_j_all
                Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
                err = kornia.geometry.symmetrical_epipolar_distance(self.corr_points_i_all, self.corr_points_j_all, Fm, squared=False, eps=1e-08)

                lepipolar = (err * self.corr_weight_all).sum() / (self.corr_mask_all * self.corr_weight_all).sum()

                loss += lepipolar * cfg.epi_loss_weight

                # Log lepipolar for visualization
                self.plot_epipolar_loss(lepipolar, step)

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if cfg.use_corres_epipolar_loss:
                desc += f"corres epipolar loss={lepipolar.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.intrinsics_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            
            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", ckpt: dict = None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if ckpt is not None:
            if "pose_adjust" in ckpt.keys() and self.cfg.pose_opt:
                if self.world_size > 1:
                    self.pose_adjust.module.load_state_dict(ckpt["pose_adjust"])
                else:
                    self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
            if "app_module" in ckpt.keys() and self.cfg.app_opt:
                if self.world_size > 1:
                    self.app_module.module.load_state_dict(ckpt["app_module"])
                else:
                    self.app_module.load_state_dict(ckpt["app_module"])

        trainloader = torch.utils.data.DataLoader(
            self.trainvalset, batch_size=1, shuffle=False, num_workers=1
        )

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        train_metrics = defaultdict(list)

        camtoworlds_est, camtoworlds_gt, intrinsics, depthmaps = [], [], [],[]
        for i, data in enumerate(trainloader):
            camtoworlds = data["camtoworld"].to(device)
            if "camtoworld_gt" in data:
                pesudo_gt = False
                camtoworld_gt = data["camtoworld_gt"].to(device)
            else:
                pesudo_gt = True
                camtoworld_gt = camtoworlds

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, data['image_id'].to(device))
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, data['image_id'].to(device))

            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # intrinsics optimization
            if cfg.intrinsics_opt:
                Ks = torch.eye(3, dtype=torch.float32, device=device)[None].expand(camtoworlds.shape[0], 3, 3).clone()
                Ks[:, 0, 0] = Ks[:, 1, 1] = self.focal_opt.exp()
                Ks[:, 0:2, 2] = self.pp_opt * self.imsize
                if step % 50 == 0:
                    print(Ks)

            camtoworlds_est.append(camtoworlds)
            camtoworlds_gt.append(camtoworld_gt)
            intrinsics.append(Ks)

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]


            depths = (depths - depths.min()) / (depths.max() - depths.min())

            if world_rank == 0:
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                train_metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                train_metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                train_metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        metrics = defaultdict(list)

        a = torch.stack(camtoworlds_est,dim=0).squeeze(1).detach().cpu().numpy()
        b = torch.stack(camtoworlds_gt,dim=0).squeeze(1).detach().cpu().numpy()
        transform = align_pose(b, a).to(device)

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld_gt"].to(device)
            camtoworlds = torch.einsum('ij,bjk->bik', transform, camtoworlds)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # intrinsics optimization
            if cfg.intrinsics_opt:
                Ks = torch.eye(3, dtype=torch.float32, device=device)[None].expand(camtoworlds.shape[0], 3, 3).clone()
                Ks[:, 0, 0] = Ks[:, 1, 1] = self.focal_opt.exp()
                Ks[:, 0:2, 2] = self.pp_opt * self.imsize

            torch.cuda.synchronize()
            tic = time.time()
            camtoworlds = self.optim_camtoworlds(camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                pixels=pixels,
                image_id=i,
            )
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]

            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [pixels, colors, (pixels-colors).abs(), depths.repeat(1,1,1,3)]

            if world_rank == 0:
                # Save rendered image
                rendered_img = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/rendered_{i:04d}.png", 
                    rendered_img
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
        
        # eval ate
        if pesudo_gt:
            print("pesudo gt!!")
        ape_stats = eval_ate(
            camtoworlds_est, 
            camtoworlds_gt,
            self.stats_dir,
            step,
            monocular=True,
        )


        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            train_stats = {k: torch.stack(v).mean().item() for k, v in train_metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"TRAIN PSNR: {train_stats['psnr']:.3f}, TRAIN SSIM: {train_stats['ssim']:.4f}, TRAIN LPIPS: {train_stats['lpips']:.3f} "
                f"TEST PSNR: {stats['psnr']:.3f}, TEST SSIM: {stats['ssim']:.4f}, TEST LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            print(f"{self.stats_dir}/train_step{step:04d}.json")
            with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                json.dump(train_stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    
    # Experimental
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l



    def plot_epipolar_loss(self, lepipolar, step):
        """Log epipolar loss to tensorboard.
        
        Args:
            lepipolar: Current epipolar loss value
            step: Current training step
        """
        # Initialize history if not exists
        if not hasattr(self, 'epipolar_loss_history'):
            self.epipolar_loss_history = []
        
        # Append current loss
        self.epipolar_loss_history.append(lepipolar.item())
        
        # Log to tensorboard every step
        self.writer.add_scalar('train/epipolar_loss', lepipolar.item(), step)
        

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step, ckpt=ckpts[0])
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def l1_dist(a, b, weight):
    return ((a - b).norm(dim=-1) * weight)

def smoothl1_dist(a, b, weight, beta=1.0):
    diff = (a - b).norm(dim=-1)
    mask = diff < beta
    smooth_l1 = torch.where(
        mask,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta
    )
    return smooth_l1 * weight

def compute_gradient_loss(pixels, colors, edge_threshold=4, rgb_boundary_threshold=0.01):
    """
    Compute gradient-aware loss with masking
    
    Args:
        pixels: Target image tensor [B, H, W, C]
        colors: Rendered image tensor [B, H, W, C] 
        edge_threshold: Threshold for edge detection relative to median gradient
        rgb_boundary_threshold: Threshold for RGB boundary detection
    """
    def image_gradient(image):
        # Compute image gradient using Scharr Filter
        c = image.shape[0]
        conv_y = torch.tensor(
            [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
        )
        conv_x = torch.tensor(
            [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
        )
        normalizer = 1.0 / torch.abs(conv_y).sum()
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        img_grad_v = normalizer * torch.nn.functional.conv2d(
            p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = normalizer * torch.nn.functional.conv2d(
            p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        return img_grad_v[0], img_grad_h[0]


    def image_gradient_mask(image, eps=0.01):
        # Compute image gradient mask
        c = image.shape[0]
        conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        p_img = torch.abs(p_img) > eps
        img_grad_v = torch.nn.functional.conv2d(
            p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = torch.nn.functional.conv2d(
            p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
        )

        return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


    # Process each batch item
    batch_losses = []
    for b in range(pixels.shape[0]):
        # Convert target image to grayscale [1, H, W]
        gray_img = pixels[b].permute(2, 0, 1).mean(dim=0, keepdim=True)
        
        # Compute gradients and masks
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        
        # Apply masks to gradients
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        
        # Compute gradient intensity
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)
        
        # Create edge mask based on median threshold
        median_img_grad_intensity = torch.median(img_grad_intensity)
        image_mask = (img_grad_intensity > median_img_grad_intensity * edge_threshold).float()
        
        # Create RGB boundary mask
        rgb_pixel_mask = (pixels[b].sum(dim=-1) > rgb_boundary_threshold).float()
        
        # Combine masks
        combined_mask = image_mask * rgb_pixel_mask

        # Compute masked L1 loss
        batch_loss = combined_mask * torch.abs(colors[b] - pixels[b]).mean(dim=-1)
        batch_losses.append(batch_loss.sum() / (combined_mask.sum() + 1e-8))

    # Average losses across batch
    return torch.stack(batch_losses).mean()

if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
