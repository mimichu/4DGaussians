exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/main_wrist_old_20K/can_main_wrist/"  &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/main_wrist_old_20K/lift_main_wrist/" &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/main_wrist_old_20K/square_main_wrist/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/main_wrist_old_20K/transport_main_wrist/"   

# export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/hellwarrior/"  &
# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name1/hook/" &
# export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/trex/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/mutant/"   &
wait
echo "Done"
