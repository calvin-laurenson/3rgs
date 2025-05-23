# Dataset root paths
TNT_ROOT="data/t2/training_colmap" # set to your path
MIP360_ROOT="data/mip360_colmap"   # set to your path
DTU_ROOT="data/DTU"                # set to your path

# TnT
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${TNT_ROOT}/Truck --data_factor 2 --result_dir results/mcmc-mlp-epi/Truck --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${TNT_ROOT}/Caterpillar --data_factor 2 --result_dir results/mcmc-mlp-epi/Caterpillar --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${TNT_ROOT}/Ignatius --data_factor 2 --result_dir results/mcmc-mlp-epi/Ignatius --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${TNT_ROOT}/Meetingroom --data_factor 2 --result_dir results/mcmc-mlp-epi/Meetingroom --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${TNT_ROOT}/Barn --data_factor 2 --result_dir results/mcmc-mlp-epi/Barn --pose_opt_type mlp --use_corres_epipolar_loss

# MipNeRF360
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/bicycle --data_factor 4 --result_dir results/mcmc-mlp-epi/bicycle --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/garden --data_factor 4 --result_dir results/mcmc-mlp-epi/garden --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/room --data_factor 2 --result_dir results/mcmc-mlp-epi/room --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/counter --data_factor 2 --result_dir results/mcmc-mlp-epi/counter --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/bonsai --data_factor 2 --result_dir results/mcmc-mlp-epi/bonsai --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/kitchen --data_factor 2 --result_dir results/mcmc-mlp-epi/kitchen --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${MIP360_ROOT}/stump --data_factor 4 --result_dir results/mcmc-mlp-epi/stump --pose_opt_type mlp --use_corres_epipolar_loss

# DTU
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${DTU_ROOT}/scan69 --data_factor 2 --result_dir results/mcmc-mlp-epi/scan69 --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${DTU_ROOT}/scan83 --data_factor 2 --result_dir results/mcmc-mlp-epi/scan83 --pose_opt_type mlp --use_corres_epipolar_loss 
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${DTU_ROOT}/scan106 --data_factor 2 --result_dir results/mcmc-mlp-epi/scan106 --pose_opt_type mlp --use_corres_epipolar_loss
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc --data_dir ${DTU_ROOT}/scan110 --data_factor 2 --result_dir results/mcmc-mlp-epi/scan110 --pose_opt_type mlp --use_corres_epipolar_loss 

