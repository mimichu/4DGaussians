exp_name1=$1
echo $exp_name1

# export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/robomimic/can_panorama_move_fov_45_dynamic --port 6300 --expname "$exp_name1/can_panorama_move_fov_45_dynamic" --configs arguments/$exp_name1/default.py &
export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/robomimic/can_panorama_move_fov_45_static_wrist --port 6300 --expname "$exp_name1/can_panorama_move_fov_45_static_wrist" --configs arguments/$exp_name1/default.py &
export CUDA_VISIBLE_DEVICES=6&&python train.py -s data/robomimic/can_panorama_move_fov_90_static_wrist --port 6301 --expname "$exp_name1/can_panorama_move_fov_90_static_wrist" --configs arguments/$exp_name1/default.py &
export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/robomimic/can_panorama_move_fov_120_static_wrist --port 6302 --expname "$exp_name1/can_panorama_move_fov_120_static_wrist" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_45_static_wrist/" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_90_static_wrist/" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/can_panorama_move_fov_120_static_wrist/" --configs arguments/$exp_name1/default.py &

# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_panorama_move_fov_45_static_wrist --port 6301 --expname "$exp_name1/can_panorama_move_fov_45_static_wrist" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_panorama_move_fov_90_static_wrist --port 6302 --expname "$exp_name1/can_panorama_move_fov_90_static_wrist" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/can_panorama_move_fov_120_static_wrist --port 6303 --expname "$exp_name1/can_panorama_move_fov_120_static_wrist" --configs arguments/$exp_name1/default.py &


## main wrist no fov set
 
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/can_main_wrist_fov_noset --port 6300 --expname "$exp_name1/can_main_wrist_fov_noset" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift_horizontal_move --port 6208 --expname "$exp_name1/lift_horizontal_move" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square_horizontal_move --port 6204 --expname "$exp_name1/square_horizontal_move" --configs arguments/$exp_name1/default.py &
 
# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can --port 6100 --expname "$exp_name1/can" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift --port 6105 --expname "$exp_name1/lift" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square --port 6104 --expname "$exp_name1/square" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/transport --port 6106 --expname "$exp_name1/transport" --configs arguments/$exp_name1/default.py &

# ## panorama
# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_panorama --port 6200 --expname "$exp_name1/can_panorama" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift_panorama --port 6208 --expname "$exp_name1/lift_panorama" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square_panorama --port 6204 --expname "$exp_name1/square_panorama" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/transport_panorama --port 6206 --expname "$exp_name1/transport_panorama" --configs arguments/$exp_name1/default.py &
# wait &&

## horizontal
# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_horizontal_move --port 6200 --expname "$exp_name1/can_horizontal_move" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift_horizontal_move --port 6208 --expname "$exp_name1/lift_horizontal_move" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square_horizontal_move --port 6204 --expname "$exp_name1/square_horizontal_move" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/transport_panorama --port 6206 --expname "$exp_name1/transport_panorama" --configs arguments/$exp_name1/default.py &
wait &&

# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_main_wrist --port 6100 --expname "$exp_name1/can_main_wrist " --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift_main_wrist  --port 6108 --expname "$exp_name1/lift_main_wrist " --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square_main_wrist  --port 6104 --expname "$exp_name1/square_main_wrist " --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/transport_main_wrist  --port 6106 --expname "$exp_name1/transport_main_wrist " --configs arguments/$exp_name1/default.py &
# wait &&

# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_one_wrist_wiggle --port 6100 --expname "$exp_name1/can_one_wrist_wiggle" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/lift_one_wrist_wiggle   --port 6108 --expname "$exp_name1/lift_one_wrist_wiggle" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/square_one_wrist_wiggle   --port 6104 --expname "$exp_name1/square_one_wrist_wiggle" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/transport_one_wrist_wiggle  --port 6110 --expname "$exp_name1/transport_one_wrist_wiggle" --configs arguments/$exp_name1/default.py &
# wait &&

# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/can_main_wrist_fov_90_sequencial --port 6111 --expname "$exp_name1/can_main_wrist_fov_90_sequencial" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/can_horizontal_move_fov_90 --port 6110 --expname "$exp_name1/can_horizontal_move_fov_90" --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/can_panorama_move_fov_90 --port 6112 --expname "$exp_name1/can_panorama_move_fov_90" --configs arguments/$exp_name1/default.py &



# render main wrist
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_main_wrist/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/lift_main_wrist/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/square_main_wrist/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/transport_main_wrist/" --skip_train --configs arguments/$exp_name1/default.py &

# render panorama
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_horizontal_move_fov_90/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/can_main_wrist_fov_90_sequencial/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/square_panorama/" --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/transport_main_wrist/" --skip_train --configs arguments/$exp_name1/default.py &




echo "Done"
