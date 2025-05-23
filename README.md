# 3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS

This repository contains the official implementation of **3R-GS**, introduced in our paper:

🌐 **Project Page**: [https://zsh523.github.io/3R-GS/](https://zsh523.github.io/3R-GS/)


> **3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS**  
> *Zhisheng Huang, Peng Wang, Jingdong Zhang, Yuan Liu, Xin Li, Wenping Wang*  
> [arXiv:2504.04294](https://arxiv.org/abs/2504.04294)

---

## 🛠 Installation

To set up the environment:

```bash
conda create --name 3rgs python=3.11 -y
conda activate 3rgs
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

---

## 📁 Data Preparation

### 1. Download datasets

Download the original datasets:

- [Tanks and Temples (TnT)](https://www.tanksandtemples.org/download/)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [DTU](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9)

### 2. Download and extract MASt3R-SfM outputs

We provide precomputed MASt3R-SfM results for each dataset:  
📁 [Google Drive](https://drive.google.com/drive/u/1/folders/1N4KSjWbp5xFhOVZQZ16_UbL60_1Sni9g)

After downloading, extract each archive to the corresponding dataset root:

```bash
tar -xzf MipNeRF360.tar.gz --strip-components=1 -C YOUR_MIPNERF360_ROOT_PATH
tar -xzf TnT.tar.gz        --strip-components=1 -C YOUR_TNT_ROOT_PATH
tar -xzf DTU.tar.gz        --strip-components=1 -C YOUR_DTU_ROOT_PATH
```

Each scene directory will have the following structure:

```
your_dataset/
└── scene/
    ├── images/               # Original RGB images
    ├── sparse/               # Original GT annotations
    ├── mast3r/               # MASt3R-SfM outputs
    ├── images_train.txt      # Training split list
    ├── images_test.txt       # Testing split list
    ├── pose_gt_train.npy     # Ground truth train poses
    └── pose_gt_test.npy      # Ground truth test poses
```

> Note: The MASt3R-SfM pipeline is not yet integrated into this codebase. For now, please refer to the [MASt3R GitHub](https://github.com/naver/mast3r/tree/mast3r_sfm) for details.

---

## 🚀 Usage

### Run training or evaluation:

```bash
python src/trainer.py <mode> \
    --data_dir <INPUT_SCENE_PATH> \
    --data_factor <IMAGE_DOWNSAMPLE_RATE> \
    --result_dir <OUTPUT_PATH> \
    --pose_opt_type <pose_mode> \
    [--use_corres_epipolar_loss | --no-use_corres_epipolar_loss] \
    [--ckpt <CHECKPOINT_PATH>]
```

### Arguments:

- `<mode>`: Pose optimization mode
  - `default`: Original 3DGS optimization
  - `mcmc`: MCMC-based 3DGS optimization

- `--data_dir`: Path to the input scene (e.g., `${TNT_ROOT}/Truck`)
- `--data_factor`: Image downsampling factor (e.g., 1, 2, 4)
- `--result_dir`: Output directory for saving results
- `--pose_opt_type`: Pose optimization method
  - `sfm`: Optimize camera poses directly
  - `mlp`: Use MLP-based global pose refinement
- `--use_corres_epipolar_loss` or `--no-use_corres_epipolar_loss`: Whether to apply global epipolar loss
- `--ckpt`: (Optional) Path to a checkpoint for evaluation

---

## 📊 Reproducing Paper Results

- **Naive joint optimization (baseline)**:
  ```bash
  bash scripts/3dgs_train.sh
  ```

- **Our method (3R-GS)**:
  ```bash
  bash scripts/3rgs_train.sh
  ```

---

## 🙏 Acknowledgements

- 3D Gaussian Splatting code is based on [gsplat](https://github.com/nerfstudio-project/gsplat)
- Evaluation scripts adapted from [MonoGS](https://github.com/muskie82/MonoGS)

---

## 📖 Citation

If you find our project helpful, please consider citing:

```bibtex
@misc{huang20253rgsbestpracticeoptimizing,
  title     = {3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS},
  author    = {Zhisheng Huang and Peng Wang and Jingdong Zhang and Yuan Liu and Xin Li and Wenping Wang},
  year      = {2025},
  eprint    = {2504.04294},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url       = {https://arxiv.org/abs/2504.04294}
}
```
