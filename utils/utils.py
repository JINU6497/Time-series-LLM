import torch
import torch.nn as nn
import numpy as np
import os
import random
import sys
import time
import logging
import glob
import shutil
import json

def set_seed(random_seed: int = 72):
    """
    set deterministic seed

    Parameters
    --------
    random_seed : int(default=72)
        random seed number

    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA Randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def load_model(resume: int, logdir: str):
    """
    load saved model
    
    Parameters
    ---------
    resume : int
        version to re-train or test
    logdir : str
        version directory to load model

    Return
    ------
    weights
        saved generator weights
    start_epoch
        last epoch of saved experiment version
    last_lr
        saved learning rate
    best_metrics
        best metrics of saved experiment version

    """
    savedir = os.path.join(logdir)
    for name in os.listdir(savedir):
        if '.pth' in name:
            modelname = name

    modelpath = os.path.join(savedir, modelname)
    print('modelpath: ', modelpath)

    loadfile = torch.load(modelpath)
    weights = loadfile['weight']
    start_epoch = loadfile['best_epoch']
    last_lr = loadfile['best_lr']
    best_metrics = loadfile['best_loss']

    return weights, start_epoch, last_lr, best_metrics



def version_build(logdir: str, is_train: bool, resume: int=None) -> str:
    """
    make n th version folder

    Parameters
    ---------
    logdir : str
        log directory
    is_train : bool
        train or not
    resume : int
        version to re-train or test

    Return
    ------
    logdir : str
        version directory to log history

    """
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    if is_train and resume is None:
        version = len(os.listdir(logdir))
    else:
        version = resume

    logdir = os.path.join(logdir, f'version{version}')

    if is_train and resume is None:
        os.makedirs(logdir)

    return logdir


last_time = time.time()
begin_time = last_time


def progress_bar(current: int, total: int, name: str, msg: str = None, width: int = None):
    """
    progress bar to show history
    
    Parameters
    ---------
    current : int
        current batch index
    total : str
        total data length
    name : str
        description of progress bar
    msg : str(default=None)
        history of training model
    width : int(default=None)
        stty size

    """

    if width is None:
        _, term_width = os.popen('stty size', 'r').read().split()
        term_width = int(term_width)
    else:
        term_width = width

    TOTAL_BAR_LENGTH = 65.

    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(f'{name} [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = ['  Step: %s' % format_time(step_time), ' | Tot: %s' % format_time(tot_time)]
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secondsf = int(seconds)
    seconds -= secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class CheckPoint:
    """
    checkpoint

    Parameters
    ----------
    logdir : str
    last_metrics :  float(default=None)

    Attributes
    ----------
    best_score
    best_epoch
    logdir

    """
    def __init__(self, logdir: str, last_metrics: float = None, metric_type: str = 'loss'):
        assert metric_type in ['loss', 'score']
        if metric_type == 'loss':
            self.best_score = np.inf if last_metrics is None else last_metrics
        else:
            self.best_score = 0 if last_metrics is None else last_metrics
        self.metric_type = metric_type
        self.best_epoch = 0
        self.logdir = logdir

    def check(self, epoch: int, model, score: float, lr: float):
        """
        Parameters
        ---------
        epoch
            current epoch
        model
            model
        score
            current score
        lr
            current learning rate
        """

        if score < self.best_score and self.metric_type == 'loss':
            self.model_save(epoch, model, score, lr)
        elif score > self.best_score and self.metric_type == 'score':
            self.model_save(epoch, model, score, lr)

    def model_save(self, epoch: int, model, score: float, lr: float):
        """
        Parameters
        ---------
        epoch
            current epoch
        model
            model
        score
            current score
        lr
            current learning rate of generator
        """

        print('Save complete, epoch: {0:}: Best loss has changed from {1:.5f} to {2:.5f}'.format(epoch, self.best_score,
                                                                                                 score))

        state = {
            'best_loss': score,
            'best_epoch': epoch,
            'best_lr': lr,
        }

        ## model save
        if isinstance(model, nn.DataParallel):
            state['weight'] = model.module.state_dict()
        else:
            state['weight'] = model.state_dict()

        save_lst = os.listdir(self.logdir)
        for f in save_lst:
            if '.pth' in f:
                os.remove(os.path.join(self.logdir, f))

        f_name = f'{epoch}.pth'
        torch.save(state, os.path.join(self.logdir, f_name))

        self.best_score = score
        self.best_epoch = epoch


def log_setting(logdir: str,
                log_name: str,
                formatter: str = '%(asctime)s|%(name)s|%(levelname)s:%(message)s'):
    """
    logger setting

    Parameters
    ----------
    log_name : str
        logger's name
    formatter : str(default=%(asctime)s|%(name)s|%(levelname)s:%(message)s)
        logging formatter

    Return
    ------
    logger

    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(logdir, f'{log_name}.log'))
    formatter = logging.Formatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def make_save(savedir: str, resume: bool = False) -> str:
    # resume
    if resume:
        assert os.path.isdir(savedir), f'{savedir} does not exist'
        # check version
        version = len([f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isdir(f)])
        # init version
        if version == 0:
            # check saved files
            files = [f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isfile(f)]
            # make version0
            version0_dir = os.path.join(savedir, f'train{version}')
            os.makedirs(version0_dir)
            # move saved files into version0
            for f in files:
                shutil.move(f, f.replace(savedir, version0_dir))
                
            version += 1
        
        savedir = os.path.join(savedir, f'train{version}')

    # make save directory
    assert not os.path.isdir(savedir), f'{savedir} already exists'
    os.makedirs(savedir)
    print("make save directory {}".format(savedir))

    return savedir

def load_resume_model(model, savedir: str, resume_num: int):
    # check latest version (previous version)
    latest_version = int(savedir.split('train')[-1]) - 1
    
    # load latest weights
    new_weights = torch.load(
        os.path.join(
            savedir.replace(f'train{latest_version+1}', f'train{resume_num}'),
            'latest_model.pt'
        )
    )
    # load weights
    model.load_state_dict(new_weights, strict=False)
    
    print('load model from (version {})'.format(resume_num))
    
class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)