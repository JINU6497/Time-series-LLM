python main.py \
    --setting 0 \
    --default_cfg ./configs/default_setting.yaml \
    --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
    --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    MODEL.modelname TMAE2 \
    DATASET.window_size 60 \
    DATASET.pred_len 15 \
    TRAIN.batch_size 4 \
    TRAIN.eval_interval 800 \
    TRAIN.log_interval 800 \
    TRAIN.log_test_interval 100 \
    DEFAULT.exp_name forecasting_exchange_rate_60_15_noscale \

python main.py \
    --setting 0 \
    --default_cfg ./configs/default_setting.yaml \
    --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
    --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    MODEL.modelname TMAE2 \
    DATASET.window_size 96 \
    DATASET.pred_len 24 \
    TRAIN.batch_size 4 \
    TRAIN.eval_interval 700 \
    TRAIN.log_interval 700 \
    TRAIN.log_test_interval 70 \
    DEFAULT.exp_name forecasting_exchange_rate_96_24_noscale \

# python main.py \
#     --setting 0 \
#     --default_cfg ./configs/default_setting.yaml \
#     --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
#     --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
#     --model_cfg ./configs/model_setting.yaml \
#     MODEL.modelname TMAE \
#     DATASET.window_size 96 \
#     DATASET.pred_len 24 \
#     DEFAULT.exp_name forecasting_exchange_rate_96_24_noscale \
#     TRAIN.batch_size 64 \
#     TRAIN.eval_interval 80 \
#     TRAIN.log_interval 80 \
#     TRAIN.log_test_interval 80 

# python main.py \
#     --setting 0 \
#     --default_cfg ./configs/default_setting.yaml \
#     --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
#     --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
#     --model_cfg ./configs/model_setting.yaml \
#     MODEL.modelname TimesNet \
#     DATASET.window_size 96 \
#     DATASET.pred_len 24 \
#     DEFAULT.exp_name forecasting_exchange_rate_96_24_noscale

# python main.py \
#     --setting 0 \
#     --default_cfg ./configs/default_setting.yaml \
#     --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
#     --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
#     --model_cfg ./configs/model_setting.yaml \
#     MODEL.modelname TimesNet \
#     DATASET.window_size 60 \
#     DATASET.pred_len 15 \
#     DEFAULT.exp_name forecasting_exchange_rate_60_15_noscale

# python main.py \
#     --setting 0 \
#     --default_cfg ./configs/default_setting.yaml \
#     --pretrain_default_cfg ./configs/pretrain_default_setting.yaml \
#     --fine_tuning_default_cfg ./configs/fine_tuning_default_setting.yaml \
#     --model_cfg ./configs/model_setting.yaml \
#     MODEL.modelname DLinear \
#     DATASET.window_size 96 \
#     DATASET.pred_len 24 \
#     DEFAULT.exp_name forecasting_exchange_rate_96_24_noscale112
