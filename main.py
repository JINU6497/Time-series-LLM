import numpy as np
import os
import wandb
import json
import shutil
import logging

from omegaconf import OmegaConf
from glob import glob

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader

from exp_builder.exp_long_term_forecasting import training_long_term_forecasting, test_long_term_forecasting
from data_provider import create_dataloader_default
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import make_save, load_resume_model, Float32Encoder
from arguments import parser
from utils.tools import update_information


_logger = get_logger('train')

def main(cfg):
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelrator
    accelerator = Accelerator()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.modelname, cfg.DEFAULT.exp_name)
    savedir = make_save(accelerator = accelerator, savedir=savedir, resume=cfg.TRAIN.resume)

    # set device
    _logger.info('Device: {}'.format(accelerator.device), main_process_only=False)

    # load and define dataloader
    information_dict, trn_dataloader, valid_dataloader, test_dataloader = create_dataloader_default(
                task_name         = cfg.DATASET.taskname,
                data_name         = cfg.DATASET.dataname,
                sub_data_name     = cfg.DATASET.sub_data_name,
                data_info         = cfg.DATAINFO,
                train_setting     = cfg.TRAIN,
                scale             = cfg.DATASET.scale,
                window_size       = cfg.DATASET.window_size,
                label_len         = cfg.DATASET.label_len,
                pred_len          = cfg.DATASET.pred_len,
                model_type        = cfg.DATASET.model_type,
                split_rate        = cfg.DATASET.split_rate,
                timeenc           = cfg.DATASET.timeenc,
                freq              = cfg.DATASET.freq
                )

    update_information(model_name        = cfg.MODEL.modelname, 
                        cfg              = cfg, 
                        information_dict = information_dict)

    # save configs
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))

    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    # # load weights
    # if cfg.TRAIN.resume:
    #     load_resume_model(model=model, savedir=savedir, resume_num=cfg.TRAIN.resume_number)

    _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    # set training
    criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
    optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name, lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)

    model, optimizer, trn_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, trn_dataloader, valid_dataloader, test_dataloader
    )
    
    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                    group   = cfg.TRAIN.wandb.exp_name,
                    project = cfg.TRAIN.wandb.project_name, 
                    entity  = cfg.TRAIN.wandb.entity, 
                    config  = OmegaConf.to_container(cfg))

    if cfg.DATASET.taskname == 'long_term_forecast':
        # fitting model
        training_long_term_forecasting(
        model                 = model, 
        trainloader           = trn_dataloader, 
        validloader           = valid_dataloader, 
        criterion             = criterion, 
        optimizer             = optimizer,
        accelerator           = accelerator, 
        epochs                = cfg.TRAIN.epoch,
        eval_epochs           = cfg.TRAIN.eval_epochs, 
        log_epochs            = cfg.TRAIN.log_epochs,
        log_eval_iter         = cfg.TRAIN.log_eval_iter,
        use_wandb             = cfg.TRAIN.wandb.use, 
        wandb_iter            = cfg.TRAIN.wandb.iter,
        ckp_metric            = cfg.TRAIN.ckp_metric, 
        label_len             = cfg.DATASET.label_len,
        pred_len              = cfg.DATASET.pred_len,
        savedir               = savedir,
        model_name            = cfg.MODEL.modelname,
        early_stopping_metric = cfg.TRAIN.early_stopping_metric,
        early_stopping_count  = cfg.TRAIN.early_stopping_count,
        lradj                 = cfg.TRAIN.lradj,
        learning_rate         = cfg.OPTIMIZER.lr,
        model_config          = cfg.MODELSETTING
        )
    
        # torch.cuda.empty_cache()

        # load best checkpoint weights
        model = accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
        model = accelerator.prepare(model)
        
        # test results
        fine_tuning_test_metrics = test_long_term_forecasting(
        accelerator   = accelerator,
        model         = model, 
        dataloader    = test_dataloader, 
        criterion     = criterion, 
        log_interval  = cfg.TRAIN.log_eval_iter, 
        label_len     = cfg.DATASET.label_len,
        pred_len      = cfg.DATASET.pred_len,
        name          = 'TEST',
        savedir       = savedir,
        model_name    = cfg.MODEL.modelname,
        model_config  = cfg.MODELSETTING,
        return_output = False
        )

    _logger.info('{} test_metrics: {}'.format(cfg.DATASET.taskname, fine_tuning_test_metrics))
    json.dump(fine_tuning_test_metrics, open(os.path.join(savedir, 
                        f'{cfg.DATASET.taskname}test_results.json'),'w'), indent='\t', cls=Float32Encoder)

if __name__=='__main__':
    cfg = parser()
    main(cfg)
