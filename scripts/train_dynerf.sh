exp_name=$1
# export CUDA_VISIBLE_DEVICES=7&&python train.py -s data/dynerf/cut_roasted_beef --port 6078 --expname "$exp_name/cut_roasted_beef" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/flame_steak --port 6025 --expname "$exp_name/cook_spinach" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dynerf/coffee_martini --port 6026 --expname "$exp_name/coffee_martini" --configs arguments/$exp_name/default.py &

# export CUDA_VISIBLE_DEVICES=6&&python train.py -s data/dynerf/cook_spinach --port 6023 --expname "$exp_name/cook_spinach" --configs arguments/$exp_name/default.py &
#export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dynerf/sear_steak --port 6024 --expname "$exp_name/sear_steak" --configs arguments/$exp_name/default.py  &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dynerf/coffee_martini --port 6030 --expname "$exp_name/coffee_martini" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/cut_roasted_beef --port 6029 --expname "$exp_name/cut_roasted_beef" --configs arguments/$exp_name/default.py &

 
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/hypernerf/interp/hand1-dense-v2 --port 6071 --expname "$exp_name/hand1-dense-v2" --configs arguments/$exp_name/hand1-dense-v2.py 
# wait
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/cut_roasted_beef --configs arguments/$exp_name/default.py --skip_train --skip_test&
# export CUDA_VISIBLE_DEVICES=6&&python render.py --model_path output/$exp_name/cook_spinach  --configs arguments/$exp_name/default.py --skip_train --skip_test &
# export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path output/$exp_name/sear_steak --configs arguments/$exp_name/default.py --skip_train --skip_test&
# export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path output/$exp_name/coffee_martini --configs arguments/$exp_name/default.py --skip_train --skip_test&

# # export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path output/$exp_name/hand1-dense-v2 --configs arguments/$exp_name/hand1-dense-v2.py --skip_train
# wait
export CUDA_VISIBLE_DEVICES=7&&python metrics.py --model_path "output/$exp_name/cut_roasted_beef/"  &
export CUDA_VISIBLE_DEVICES=6&&python metrics.py --model_path "output/$exp_name/cook_spinach/" &
export CUDA_VISIBLE_DEVICES=5&&python metrics.py --model_path "output/$exp_name/sear_steak/" &
# export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name/coffee_martini/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/hand1-dense-v2/" 
wait
echo "Done"