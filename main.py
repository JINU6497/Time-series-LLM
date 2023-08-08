import numpy as np
import os
import wandb
import json
import shutil
import logging
from glob import glob

import torch
from torch.utils.data import DataLoader

from exp_builder.exp_long_term_forecasting import training_long_term_forecasting, test_long_term_forecasting
from exp_builder.exp_anomaly_detection import training_anomaly_detection, test_anomaly_detection

from data_provider import create_dataloader_default
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import set_seed, make_save, load_resume_model, Float32Encoder
from arguments import parser

from accelerate import Accelerator
from omegaconf import OmegaConf

_logger = logging.getLogger('train')

def main(cfg):
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelrator
    accelerator = Accelerator()
    
    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.modelname, cfg.DEFAULT.exp_name)
    savedir = make_save(savedir=savedir, resume=cfg.TRAIN.resume)
    
    # save configs
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    setup_default_logging()
    
    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load and define dataloader
    trn_dataloader, valid_dataloader, test_dataloader = create_dataloader_default(
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

    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    # load weights
    # if cfg.TRAIN.resume:
        # load_resume_model(model=model, savedir=savedir, resume_num=cfg.TRAIN.resume_number)

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

    # fitting model
    if cfg.DATASET.taskname == 'long_term_forecast':
        # fitting model
        training_long_term_forecasting(
        model              = model, 
        trainloader        = trn_dataloader, 
        validloader        = valid_dataloader, 
        criterion          = criterion, 
        optimizer          = optimizer,
        accelerator        = accelerator, 
        epochs             = cfg.TRAIN.epoch,
        eval_interval      = cfg.TRAIN.eval_interval, 
        log_interval       = cfg.TRAIN.log_interval,
        log_test_interval  = cfg.TRAIN.log_test_interval,
        use_wandb          = cfg.TRAIN.wandb.use, 
        ckp_metric         = cfg.TRAIN.ckp_metric, 
        label_len          = cfg.DATASET.label_len,
        pred_len           = cfg.DATASET.pred_len,
        savedir            = savedir,
        model_name         = cfg.MODEL.modelname
        )
        
        # torch.cuda.empty_cache()
        
        # load best checkpoint weights
        model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
        
        # test results
        fine_tuning_test_metrics = test_long_term_forecasting(
        model         = model, 
        dataloader    = test_dataloader, 
        criterion     = criterion, 
        log_interval  = cfg.TRAIN.log_test_interval, 
        label_len     = cfg.DATASET.label_len,
        pred_len      = cfg.DATASET.pred_len,
        name          = 'TEST',
        savedir       = savedir,
        model_name    = cfg.MODEL.modelname,
        return_output = False
        )
    
    elif cfg.DATASET.taskname == 'anomaly_detection':
        # fitting model
        training_anomaly_detection(
        model              = model, 
        trainloader        = trn_dataloader, 
        validloader        = valid_dataloader, 
        criterion          = criterion, 
        optimizer          = optimizer,
        accelerator        = accelerator, 
        epochs             = cfg.TRAIN.epoch,
        eval_interval      = cfg.TRAIN.eval_interval, 
        log_interval       = cfg.TRAIN.log_interval,
        log_test_interval  = cfg.TRAIN.log_test_interval,
        use_wandb          = cfg.TRAIN.wandb.use, 
        ckp_metric         = cfg.TRAIN.ckp_metric, 
        label_len          = cfg.DATASET.label_len,
        pred_len           = cfg.DATASET.pred_len,
        model_name         = cfg.DATASET.modelname,
        model_type         = cfg.DATASET.model_type,
        savedir            = savedir
        )
        
        # torch.cuda.empty_cache()
        
        # load best checkpoint weights
        model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
        
        # test results
        fine_tuning_test_metrics = test_anomaly_detection(
        model         = model, 
        dataloader    = test_dataloader, 
        criterion     = criterion, 
        log_interval  = cfg.TRAIN.log_test_interval, 
        label_len     = cfg.DATASET.label_len,
        pred_len      = cfg.DATASET.pred_len,
        model_type    = cfg.DATASET.model_type,
        model_name    = cfg.DATASET.modelname,
        name          = 'TEST',
        savedir       = savedir,
        return_output = False
        )
        
    
    _logger.info('{} test_metrics: {}'.format(cfg.DATASET.taskname, fine_tuning_test_metrics))
    json.dump(fine_tuning_test_metrics, open(os.path.join(savedir, 
                        f'{cfg.DATASET.taskname}test_results.json'),'w'), indent='\t', cls=Float32Encoder)

if __name__=='__main__':
    cfg = parser()
    main(cfg)
