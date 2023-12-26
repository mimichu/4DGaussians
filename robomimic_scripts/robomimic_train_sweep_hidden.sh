exp_name1=$1
echo $exp_name1

export CUDA_VISIBLE_DEVICES=6&&python train.py -s data/robomimic/can_fov_90_dynamic_train_static_dynamic_test_dynamic --port 6521 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_hidden" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden.py &
# export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6522 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_hidden_timesmooth" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden_timesmooth.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6523 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_hidden_netW" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden_netW.py &
# export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6524 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_hidden_multi" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden_multi.py &
# export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6525 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_hidden_l1_time" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden_l1_time.py &
# export CUDA_VISIBLE_DEVICES=6&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6527 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_hidden_defor" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_hidden_defor.py &
# export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6526 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi18K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_18K.py &

wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_hidden/" --configs arguments/$exp_name1/default_hidden.py &
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_timesmooth/" --configs arguments/$exp_name1/default_hidden_timesmooth.py &
# export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_netW/" --configs arguments/$exp_name1/default_hidden_netW.py &
# export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_multi/" --configs arguments/$exp_name1/default_hidden_multi.py &
# export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_l1_time/" --configs arguments/$exp_name1/default_hidden_l1_time.py &
# export CUDA_VISIBLE_DEVICES=6&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_defor/" --configs arguments/$exp_name1/default_hidden_defor.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi18K/" --configs arguments/$exp_name1/default_pi_18K.py &

wait 

export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_hidden/" &
# export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_timesmooth/" &
# export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_netW/" &
# export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_multi/" &
# export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_l1_time/" &
# export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_hidden_defor/" &
# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi18K/" &
echo "Done"
