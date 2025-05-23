# Dataset root paths
TNT_ROOT="data/t2/training_colmap" # set to your path
MIP360_ROOT="data/mip360_colmap"   # set to your path
DTU_ROOT="data/DTU"                # set to your path

# TnT
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${TNT_ROOT}/Truck --data_factor 2 --result_dir results/3dgs/Truck --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${TNT_ROOT}/Caterpillar --data_factor 2 --result_dir results/3dgs/Caterpillar --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${TNT_ROOT}/Ignatius --data_factor 2 --result_dir results/3dgs/Ignatius --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${TNT_ROOT}/Meetingroom --data_factor 2 --result_dir results/3dgs/Meetingroom --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${TNT_ROOT}/Barn --data_factor 2 --result_dir results/3dgs/Barn --pose_opt_type sfm --no-use_corres_epipolar_loss

# MipNeRF360
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/bicycle --data_factor 4 --result_dir results/3dgs/bicycle --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/garden --data_factor 4 --result_dir results/3dgs/garden --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/room --data_factor 2 --result_dir results/3dgs/room --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/counter --data_factor 2 --result_dir results/3dgs/counter --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/bonsai --data_factor 2 --result_dir results/3dgs/bonsai --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/kitchen --data_factor 2 --result_dir results/3dgs/kitchen --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${MIP360_ROOT}/stump --data_factor 4 --result_dir results/3dgs/stump --pose_opt_type sfm --no-use_corres_epipolar_loss

# DTU
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${DTU_ROOT}/scan69 --data_factor 2 --result_dir results/3dgs/scan69 --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${DTU_ROOT}/scan83 --data_factor 2 --result_dir results/3dgs/scan83 --pose_opt_type sfm --no-use_corres_epipolar_loss 
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${DTU_ROOT}/scan106 --data_factor 2 --result_dir results/3dgs/scan106 --pose_opt_type sfm --no-use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default --data_dir ${DTU_ROOT}/scan110 --data_factor 2 --result_dir results/3dgs/scan110 --pose_opt_type sfm --no-use_corres_epipolar_loss 

