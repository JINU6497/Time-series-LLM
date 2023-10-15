window_size=96

data_path=./dataset/weather/weather.csv
data_name=weather
model_name=LLM4TS

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size 32 \
    MODELSETTING.mode SFT

data_path=./dataset/exchange_rate/exchange_rate.csv
data_name=exchange_rate
model_name=LLM4TS

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size 64 \
    MODELSETTING.mode SFT
    
data_path=./dataset/ETT-small/ETTh1.csv
data_name=ETTh1
model_name=LLM4TS

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size 64 \
    MODELSETTING.mode SFT

data_path=./dataset/ETT-small/ETTm1.csv
data_name=ETTm1
model_name=LLM4TS

accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size 64 \
    MODELSETTING.mode SFT