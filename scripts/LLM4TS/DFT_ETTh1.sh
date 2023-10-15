window_size=96
data_path=./dataset/ETT-small/ETTh1.csv
data_name=ETTh1
model_name=LLM4TS
batch_size=64

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 96 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 192 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 336 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 720 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT