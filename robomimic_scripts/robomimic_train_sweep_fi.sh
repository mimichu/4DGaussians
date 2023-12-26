exp_name1=$1
echo $exp_name1
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6421 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_fi40K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_fi_40K.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6422 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_fi80K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_fi_80K.py &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6523 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_fi400K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_fi_400K.py &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6524 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_fi150K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_fi_150K.py &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_panorama_move_fov_90_dynamic --port 6525 --expname "$exp_name1/can_panorama_move_fov_90_dynamic_fi200K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_fi_200K.py &
wait
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi40K/" --configs arguments/$exp_name1/default_fi_40K.py &
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi80K/" --configs arguments/$exp_name1/default_fi_80K.py &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi400K/" --configs arguments/$exp_name1/default_fi_400K.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi150K/" --configs arguments/$exp_name1/default_fi_150K.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi200K/" --configs arguments/$exp_name1/default_fi_200K.py &
wait 

# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi40K/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi80K/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi400K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi150K/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/can_panorama_move_fov_90_dynamic_fi200K/" &

echo "Done"
