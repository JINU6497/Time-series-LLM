window_size=96
data_path=./dataset/weather/weather.csv
data_name=weather
model_name=LLM4TS
batch_size=32

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 96 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 192 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 336 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATAINFO.datadir $data_path \
    DATASET.pred_len 720 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.mode DFT