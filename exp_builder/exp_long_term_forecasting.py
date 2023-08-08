import logging
import wandb
import time
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from accelerate import Accelerator

from utils.metrics import cal_forecast_metric
from utils.utils import Float32Encoder

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def training_long_term_forecasting(
    model, trainloader, validloader, criterion, optimizer, accelerator: Accelerator, 
    epochs: int, eval_interval: int, log_interval: int, log_test_interval: int, use_wandb: bool, 
    ckp_metric: str, savedir: str, pred_len: int, label_len: int, model_name: str):

    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # set mode
    model.train()
    optimizer.zero_grad()
    end_time = time.time()
    
    # init best score and step
    best_score = np.inf
    
    for epoch in range(epochs):
        epoch_time = time.time()
        for idx, item in enumerate(trainloader):
            data_time_m.update(time.time() - end_time)
            
            input_ts = item['given']['ts'].type(torch.float)
            input_timeembedding = item['given']['timeembedding'].type(torch.float)
            target_ts = item['target']['ts'].type(torch.float)
            target_timeembedding = item['target']['timeembedding'].type(torch.float)

            # Output
            dec_inp = torch.zeros_like(target_ts[:, -pred_len:, :]).type(torch.float)
            dec_inp = torch.cat([target_ts[:, :label_len, :], dec_inp], dim=1).type(torch.float)
            
            if model_name == 'TMAE2':
                outputs, total_input, total_masked_output  = model(input_ts,
                                                                input_timeembedding,
                                                                dec_inp,
                                                                target_timeembedding)
                
                outputs = outputs[:, -pred_len:, :]
                target_ts = target_ts[:, -pred_len:, :]     

                # final loss
                loss_original = criterion(outputs, target_ts)

                loss_representation_list = []
                for i in range(len(total_input)):
                    loss_representation = nn.MSELoss()(total_input[i], total_masked_output[i])
                    loss_representation_list.append(loss_representation)
                loss_representation = torch.mean(torch.stack(loss_representation_list))
                loss = loss_original + loss_representation
                
            else:
                outputs = model(input_ts,
                                input_timeembedding,
                                dec_inp,
                                target_timeembedding) 
                
                outputs = outputs[:, -pred_len:, :]
                target_ts = target_ts[:, -pred_len:, :]           
                # final loss
                loss = criterion(outputs, target_ts)
            
            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item(), n = target_ts.size(0))
            
            # batch time
            batch_time_m.update(time.time() - end_time)
            
            if (idx+1) % log_interval == 0:
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(trainloader), 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = input_ts.size(0) / batch_time_m.val,
                            rate_avg   = input_ts.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                if use_wandb:
                    train_results = OrderedDict([
                        ('lr',optimizer.param_groups[0]['lr']),
                        ('train_loss',losses_m.avg)
                    ])
                    wandb.log(train_results, step=idx+1)
                    
            if (idx+1) % eval_interval == 0:
                eval_metrics = test_long_term_forecasting(
                    model         = model, 
                    dataloader    = validloader, 
                    criterion     = criterion,
                    name          = 'VALID',
                    log_interval  = log_test_interval,
                    label_len     = label_len,
                    pred_len      = pred_len,
                    return_output = False,
                    savedir       = savedir,
                    model_name    = model_name
                    )
                
                model.train()
                
                # eval results
                eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
                
                # wandb
                if use_wandb:
                    wandb.log(eval_results, step=idx+1)
                   
                # check_point
                if best_score > eval_metrics[ckp_metric]:
                    # save results
                    state = {'best_epoch':epoch ,
                             'best_step':idx+1, 
                             f'best_{ckp_metric}':eval_metrics[ckp_metric]}
                    
                    print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                        to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
                    
                    state.update(eval_results)
                    json.dump(state, open(os.path.join(savedir, f'forecasting_best_results.json'),'w'), 
                              indent='\t', cls=Float32Encoder)

                    # save model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))

                    best_score = eval_metrics[ckp_metric]
                    
            end_time = time.time()
            
    # logging best score and step
    _logger.info('Best Metric: {0:6.6f} (step {1:})\n'.format(state[f'best_{ckp_metric}'], state['best_step']))
    
    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))
    print('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
        to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

    # save latest results
    state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
    state.update(eval_results)
    json.dump(state, open(os.path.join(savedir, f'forecasting_latest_results.json'),'w'), indent='\t', cls=Float32Encoder)


def test_long_term_forecasting(model, dataloader, criterion, 
                                log_interval: int, pred_len: int, label_len: int, savedir: str, model_name: str, 
                                name: str = 'TEST', return_output: bool = False) -> dict:
    _logger.info('\n[Start {}]'.format(name))
    
    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # targets and outputs
    total_targets = []
    total_outputs = []
    
    end_time = time.time()
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            data_time_m.update(time.time() - end_time)
            # Output
            input_ts = item['given']['ts'].type(torch.float)
            input_timeembedding = item['given']['timeembedding'].type(torch.float)
            target_ts = item['target']['ts'].type(torch.float)
            target_timeembedding = item['target']['timeembedding'].type(torch.float)
            
            # Output
            dec_inp = torch.zeros_like(target_ts[:, -pred_len:, :]).type(torch.float)
            dec_inp = torch.cat([target_ts[:, :label_len, :], dec_inp], dim=1).type(torch.float)
            
            if model_name == 'TMAE2':
                outputs, total_input, total_masked_output  = model(input_ts,
                                                                input_timeembedding,
                                                                dec_inp,
                                                                target_timeembedding)
                
                outputs = outputs[:, -pred_len:, :]
                target_ts = target_ts[:, -pred_len:, :]     
                
                # final loss
                loss_original = criterion(outputs, target_ts)

                loss_representation_list = []
                for i in range(len(total_input)):
                    loss_representation = nn.MSELoss()(total_input[i], total_masked_output[i])
                    loss_representation_list.append(loss_representation)
                loss_representation = torch.mean(torch.stack(loss_representation_list))
                loss = loss_original + loss_representation
                
            else:
                outputs = model(input_ts,
                                input_timeembedding,
                                dec_inp,
                                target_timeembedding) 
                
                outputs = outputs[:, -pred_len:, :]
                target_ts = target_ts[:, -pred_len:, :]           
                # final loss
                loss = criterion(outputs, target_ts)
                
            losses_m.update(loss.item(), n=target_ts.size(0))
                        
            outputs = outputs.detach().cpu().numpy()
            target_ts = target_ts.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_targets.append(target_ts)
            
            # batch time        
            batch_time_m.update(time.time() - end_time)
            
            if (idx+1) % log_interval == 0:
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(dataloader), 
                            loss       = losses_m, 
                            batch_time = batch_time_m,
                            rate       = input_ts.size(0) / batch_time_m.val,
                            rate_avg   = input_ts.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                
            end_time = time.time()
    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    print(f'{name} shape:', total_outputs.shape, total_targets.shape)

    metrics = cal_forecast_metric(pred = total_outputs, true = total_targets)
    
    np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
    np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
    
    # logging metrics
    _logger.info("\n{}: ".format(name) + " ".join('{}: {}'.format(k.upper(), m) for k, m in metrics.items()) + "\n")
    
    # return results
    results = OrderedDict([('loss',losses_m.avg)])
    results.update([(f'{k}', m) for k, m in metrics.items()])
    
    _logger.info('[End {}]\n'.format(name))
    
    if return_output:
        return pd.DataFrame({
            'mapping_id' : dataloader.dataset.mapping_id_list,
            'target'     : total_targets,
            'predict'    : total_outputs
        })
    else:
        return results