cd src/neugrasp || exit
CUDA_VISIBLE_DEVICES=$1 /path/to/python run_training.py \
                        --cfg configs/neugrasp.yaml
cd - || exit