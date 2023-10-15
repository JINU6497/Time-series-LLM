import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')

def update_information(model_name, cfg, information_dict):
    cfg.MODELSETTING.window_size = cfg.DATASET.window_size
    cfg.MODELSETTING.label_len = cfg.DATASET.label_len
    cfg.MODELSETTING.pred_len = cfg.DATASET.pred_len
    cfg.MODELSETTING.taskname = cfg.DATASET.taskname
    cfg.MODELSETTING.pretrain = cfg.DATASET.pretrain
    cfg.MODELSETTING.timeenc = cfg.DATASET.timeenc
    cfg.MODELSETTING.freq = cfg.DATASET.freq
    cfg.MODELSETTING.embed_type = cfg.DATASET.embed_type

    if model_name == 'DLinear':
        cfg.MODELSETTING.enc_in = information_dict['enc_in']

    elif model_name == 'TMAE1':
        cfg.MODELSETTING.enc_in = information_dict['enc_in']
        cfg.MODELSETTING.dec_in = information_dict['dec_in']
        cfg.MODELSETTING.c_out = information_dict['c_out']

    elif model_name == 'TimesNet':
        cfg.MODELSETTING.enc_in = information_dict['enc_in']
        cfg.MODELSETTING.dec_in = information_dict['dec_in']
        cfg.MODELSETTING.c_out = information_dict['c_out']
    
    else:
        cfg.MODELSETTING.enc_in = information_dict['enc_in']
        cfg.MODELSETTING.dec_in = information_dict['dec_in']
        cfg.MODELSETTING.c_out = information_dict['c_out']
 

def adjust_learning_rate(optimizer, epoch, lradj, learning_rate):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate*0.1}
    elif lradj == 'type4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate*0.1}
    elif lradj == 'type5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate*0.1}
    elif lradj == 'type6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
def check_graph(xs, att, piece=1, threshold=None):
    """
    anomaly score and anomaly label visualization

    Parameters
    ----------
    xs : np.ndarray
        anomaly scores
    att : np.ndarray
        anomaly labels
    piece : int
        number of figures to separate
    threshold : float(default=None)
        anomaly threshold

    Return
    ------
    fig : plt.figure
    """
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        axs[i].plot(xticks, xs[L:R], color='#0C090A')
        ymin, ymax = axs[i].get_ylim()
        ymin = 0
        axs[i].set_ylim(ymin, ymax)
        if len(xs[L:R]) > 0:
            axs[i].vlines(xticks[np.where(att[L:R] == 1)], ymin=ymin, ymax=ymax, color='#FED8B1',
                          alpha=0.6, label='true anomaly')
        axs[i].plot(xticks, xs[L:R], color='#0C090A', label='anomaly score')
        if threshold is not None:
            axs[i].axhline(y=threshold, color='r', linestyle='--', alpha=0.8, label=f'threshold:{threshold:.4f}')
        axs[i].legend()

    return fig



def check_log_graph(xs, att, piece=1, threshold=None):
    """
    anomaly score and anomaly label visualization

    Parameters
    ----------
    xs : np.ndarray
        anomaly scores
    att : np.ndarray
        anomaly labels
    piece : int
        number of figures to separate
    threshold : float(default=None)
        anomaly threshold

    Return
    ------
    fig : plt.figure
    """
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        axs[i].plot(xticks, np.log(xs[L:R]), color='#0C090A')
        ymin, ymax = axs[i].get_ylim()
        ymin = 0
        

        axs[i].set_ylim(ymin, ymax)
        if len(xs[L:R]) > 0:
            axs[i].vlines(xticks[np.where(att[L:R] == 1)], ymin=ymin, ymax=ymax, color='#FED8B1',
                          alpha=0.6, label='true anomaly')
        axs[i].plot(xticks, np.log(xs[L:R]), color='#0C090A', label='anomaly score')
        if threshold is not None:
            axs[i].axhline(y=np.log(threshold), color='r', linestyle='--', alpha=0.8, label=f'threshold:{threshold:.4f}')
        axs[i].legend()
        
    return fig