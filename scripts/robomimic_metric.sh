exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_fov_45_static_train_pano_wrist_test_wrist_all/"  &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_fov_45_static_train_pano_wrist_test_wrist_all_30K_bs_4/"  &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_fov_45_static_train_pano_wrist_test_wrist_all_60K_bs_16/"  &

# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_45_static_wrist_30K_iters_bs_4/"  &
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_45_static_wrist_20K_iters/"  &
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_static_wrist_30K_iters_bs_4/"  &
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_static_wrist_20K_iters/"  &
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_120_static_wrist_30K_iters_bs_4/"  &
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_120_static_wrist_20K_iters/"  &

# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_static_wrist/" &
# export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_120_static_wrist/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_45_static_wrist_20K_iters/"   
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_static_wrist_20K_iters/"   
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_120_static_wrist_20K_iters/"   
# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/hellwarrior/"  &
# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/hook/" &
# export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/trex/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/mutant/"   &
wait
echo "Done"
