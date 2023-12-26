exp_name1=$1
obj_dir='can_fov_90_dynamic_train_static_dynamic_test_dynamic'
echo $exp_name1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/robomimic/$obj_dir --port 6321 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci6K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_ci_6K.py &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/robomimic/$obj_dir --port 6322 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci12K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_ci_12K.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/robomimic/$obj_dir --port 6323 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci15K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_ci_15K.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/robomimic/$obj_dir --port 6324 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci18K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_ci_18K.py &
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/robomimic/$obj_dir --port 6325 --expname "$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci1K" --configs /local/real/chuerpan/repo/4DGaussians/arguments/$exp_name1/default_ci_1K.py &
wait
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci6K/" --configs arguments/$exp_name1/default_ci_6K.py &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci12K/" --configs arguments/$exp_name1/default_ci_12K.py &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci15K/" --configs arguments/$exp_name1/default_ci_15K.py &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci18K/" --configs arguments/$exp_name1/default_ci_18K.py &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci1K/" --configs arguments/$exp_name1/default_ci_1K.py &
wait 

export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci6K/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci12K/" &
export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci15K/" &
export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci18K/" &
export CUDA_VISIBLE_DEVICES=4&&python metrics.py --model_path "output/$exp_name1/can_fov_90_dynamic_train_static_dynamic_test_dynamic_ci1K/" &

echo "Done"
