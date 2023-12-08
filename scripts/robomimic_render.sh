exp_name1=$1

# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can/"  --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/square/"  --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/lift/"  --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/lego/"  --skip_train --configs arguments/$exp_name1/lego.py  &
# wait
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/hellwarrior/"  --skip_train --configs arguments/$exp_name1/hellwarrior.py  &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/hook/"  --skip_train --configs arguments/$exp_name1/hook.py  &
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/trex/"  --skip_train --configs arguments/$exp_name1/trex.py  &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/mutant/"  --skip_train --configs arguments/$exp_name1/mutant.py   &
# wait

# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_one_wrist_wiggle/"  --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/square_one_wrist_wiggle/"  --skip_train --configs arguments/$exp_name1/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/lift_one_wrist_wiggle/"  --skip_train --configs arguments/$exp_name1/default.py &
# # export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/$exp_name1/lego_one_wrist_wiggle/"  --skip_train --configs arguments/$exp_name1/lego.py  &


export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/can_main_wrist/"  --skip_train --configs arguments/$exp_name1/default.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/$exp_name1/square_main_wrist/"  --skip_train --configs arguments/$exp_name1/default.py &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/$exp_name1/lift_main_wrist/"  --skip_train --configs arguments/$exp_name1/default.py &
echo "Done"
