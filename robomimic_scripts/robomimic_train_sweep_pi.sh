exp_name1=$1
echo $exp_name1

export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6521 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi4K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_4K.py &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6522 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi6K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_6K.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6523 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi10K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_10K.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6524 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi12K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_12K.py &
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6525 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi14K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_14K.py &
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6527 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi16K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_16K.py &
export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6526 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_pi18K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_pi_18K.py &

wait
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi4K/" --configs arguments/$exp_name1/default_pi_4K.py &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi6K/" --configs arguments/$exp_name1/default_pi_6K.py &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi10K/" --configs arguments/$exp_name1/default_pi_10K.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi12K/" --configs arguments/$exp_name1/default_pi_12K.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi14K/" --configs arguments/$exp_name1/default_pi_14K.py &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi16K/" --configs arguments/$exp_name1/default_pi_16K.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi18K/" --configs arguments/$exp_name1/default_pi_18K.py &

wait 

export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi4K/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi6K/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi10K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi12K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi14K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi16K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_pi18K/" &
echo "Done"
