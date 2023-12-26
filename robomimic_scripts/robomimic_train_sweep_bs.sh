exp_name1=$1
obj_dir='can_fov_90_dynamic_train_static_dynamic_test_dynamic'
echo $exp_name1

export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/$obj_dir --port 6311 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs1" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs1.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/$obj_dir --port 6312 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs4" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs4.py &
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/robomimic/$obj_dir --port 6313 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs8" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs8.py &
export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/$obj_dir --port 6614 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs16" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs16.py &
export CUDA_VISIBLE_DEVICES=6&&python train.py -s data/robomimic/$obj_dir --port 6315 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs32" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs32.py &
export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/robomimic/$obj_dir --port 6316 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs64" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs64.py &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/$obj_dir --port 6317 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs128" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_bs128.py &
wait 
# export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview --port 6313 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16_kplane_hd" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/iterate_kplane_highres.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview --port 6315 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16_40K_coarse" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/iterate_longer_coarse.py &


# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview --port 6316 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_30K_bs_4" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview --port 6315 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/iterate.py &

# export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_all_cam_pose_fix --port 6317 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_all_cam_pose_fix_30K_bs_4" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_all_cam_pose_fix --port 6318 --expname "$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_all_cam_pose_fix_60K_bs_16" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/iterate.py &



# export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_100K_bs_16/" --skip_train --configs arguments/$exp_name1/iterate_tune.py &
# export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16_kplane_hd/" --configs arguments/$exp_name1/iterate_kplane_highres.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16_40K_coarse/" --configs arguments/$exp_name1/iterate_longer_coarse.py &

# export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_30K_bs_4/" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_pano_wrist_test_wrist_multiview_60K_bs_16/" --configs arguments/$exp_name1/iterate.py &

export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs1/" --configs arguments/$exp_name1/default_bs1.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs4/" --configs arguments/$exp_name1/default_bs4.py &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs8/" --configs arguments/$exp_name1/default_bs8.py &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs16/" --configs arguments/$exp_name1/default_bs16.py &
export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs32/" --configs arguments/$exp_name1/default_bs32.py &
export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs64/" --configs arguments/$exp_name1/default_bs64.py &
export CUDA_VISIBLE_DEVICES=6&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs128/" --configs arguments/$exp_name1/default_bs128.py &

wait 

export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs1/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs4/" &
export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs8/" &
export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs16/" &
export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs32/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs64/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_bs128/" &

echo "Done"
