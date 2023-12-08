exp_name=$1
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/hypernerf/virg/americano --port 6168 --expname "$exp_name/americano" --configs arguments/$exp_name/default.py &
export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/hypernerf/virg/split-cookie --port 6166 --expname "$exp_name/split-cookie" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/virg/oven-mitts --port 6169 --expname "$exp_name/oven-mitts" --configs arguments/$exp_name/default.py  &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/hypernerf/virg/cut-lemon1 --port 6170 --expname "$exp_name/cut-lemon1" --configs arguments/$exp_name/default.py &

# export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/virg/broom2 --port 6168 --expname "$exp_name/broom2" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/virg/vrig-3dprinter --port 6166 --expname "$exp_name/3dprinter" --configs arguments/$exp_name/default.py &
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/virg/vrig-peel-banana --port 6169 --expname "$exp_name/peel-banana" --configs arguments/$exp_name/default.py  &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/hypernerf/virg/vrig-chicken --port 6170 --expname "$exp_name/vrig-chicken" --configs arguments/$exp_name/default.py &
# # export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/hypernerf/interp/hand1-dense-v2 --port 6071 --expname "$exp_name/hand1-dense-v2" --configs arguments/$exp_name/hand1-dense-v2.py 
# # wait

# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/americano --configs arguments/$exp_name/default.py --skip_train &
# export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path output/$exp_name/split-cookie  --configs arguments/$exp_name/default.py --skip_train &
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/oven-mitts --configs arguments/$exp_name/default.py --skip_train&
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/cut-lemon1  --configs arguments/$exp_name/default.py --skip_train&
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/broom2 --configs arguments/$exp_name/default.py --skip_train &
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/$exp_name/3dprinter  --configs arguments/$exp_name/default.py --skip_train &
# export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/$exp_name/vrig-peel-banana --configs arguments/$exp_name/default.py --skip_train&
# export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path output/$exp_name/vrig-chicken  --configs arguments/$exp_name/default.py --skip_train&

# export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path output/$exp_name/hand1-dense-v2 --configs arguments/$exp_name/hand1-dense-v2.py --skip_train
wait
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/broom2/"  &
# export CUDA_VISIBLE_DEVICES=5&&python metrics.py --model_path "output/$exp_name/3dprinter/" &
# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name/vrig-peel-banana/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/vrig-chicken/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/cut-lemon1 /"  &
# export CUDA_VISIBLE_DEVICES=5&&python metrics.py --model_path "output/$exp_name/oven-mitts/" &
# export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/$exp_name/split-cookie/" &
# export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/americano/" &
# # export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/$exp_name/hand1-dense-v2/" 
# wait
echo "Done"