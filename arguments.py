from omegaconf import OmegaConf
import argparse

def parser():
    parser = argparse.ArgumentParser(description='Time-series representation framework')
    parser.add_argument('--setting', type=int, help='0: default, 1: pretrain, 2: finetuning')
    parser.add_argument('--default_cfg', type=str, help='configuration for default setting')
    parser.add_argument('--pretrain_default_cfg', type=str, help='configuration for pretrain default setting')
    parser.add_argument('--fine_tuning_default_cfg', type=str, help='configuration for fine tuning default setting')
    parser.add_argument('--model_cfg', type=str, help='configuration for model')    
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load configs
    if args.setting == 0:
        cfg = OmegaConf.load(args.default_cfg)
        print('default setting')
    elif args.setting == 1:
        cfg = OmegaConf.load(args.pretrain_default_cfg)
        print('pretrain setting')
    elif args.setting == 2:
        cfg = OmegaConf.load(args.fine_tuning_default_cfg)
        print('fine tuning setting')

    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        try:
            # types except for int, float, and str
            OmegaConf.update(cfg, k, eval(v), merge=True)
        except:
            OmegaConf.update(cfg, k, v, merge=True)

    if args.model_cfg:
        model_cfg = OmegaConf.load(args.model_cfg)
        modelname = cfg.MODEL.modelname

        # Check if the modelname is in the model_config
        if modelname in model_cfg:
            model_setting_conf = OmegaConf.create(model_cfg[modelname])
            # Merge the specific model config with the default config
            cfg = OmegaConf.merge(cfg, {'MODELSETTING' : model_setting_conf})
        else:
            print(f"Model '{modelname}' not found in the model_config.")
            return None

    cfg.MODELSETTING.window_size = cfg.DATASET.window_size
    cfg.MODELSETTING.label_len = cfg.DATASET.label_len
    cfg.MODELSETTING.pred_len = cfg.DATASET.pred_len
    cfg.MODELSETTING.taskname = cfg.DATASET.taskname
    cfg.MODELSETTING.pretrain = cfg.DATASET.pretrain
    cfg.MODELSETTING.timeenc = cfg.DATASET.timeenc
    cfg.MODELSETTING.freq = cfg.DATASET.freq

    print(OmegaConf.to_yaml(cfg))

    return cfg  