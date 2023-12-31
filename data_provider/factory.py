import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_provider.dataset import load_anomaly_dataset, load_custom_dataset
from data_provider.build_dataset import BuildDataset_default
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

def create_dataloader_default(
        task_name: str,
        data_name: str,
        sub_data_name: str,
        data_info: dict,
        train_setting: dict,
        scale: str           = None,
        window_size: int     = 60,
        label_len: int       = 0,
        pred_len: int        = 15,
        model_type: str      = 'reconstruction',
        split_rate: list     = [0.8, 0.1, 0.1],
        timeenc: int         = 0,
        freq: str            = 'h'
        ):

    assert scale in (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')
    assert model_type in ('reconstruction', 'forecasting')
    label = None
    
    if task_name == 'anomaly_detection':
        trn, trn_ts, val, val_ts, tst, tst_ts, label = load_anomaly_dataset(dataname=data_name,
                                                                            datainfo=data_info,
                                                                            subdataname=sub_data_name)
    else:
        trn, trn_ts, val, val_ts, tst, tst_ts= load_custom_dataset(
                                                    dataname     = data_name,
                                                    datainfo     = data_info,
                                                    split_rate   = split_rate,
                                                    timeenc      = timeenc,
                                                    freq         = freq,
                                                    window_size  = window_size,
                                                    label_len    = label_len,
                                                    pred_len     = pred_len 
                                                    )

    # scaling (minmax, minmax square, minmax m1p1, standard)
    if scale is not None:
        if scale == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(trn)
            trn = scaler.transform(trn)
            val = scaler.transform(val)
            tst = scaler.transform(tst)
        elif scale == 'minmax_square':
            scaler = MinMaxScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn) ** 2
            val = scaler.transform(val) ** 2
            tst = scaler.transform(tst) ** 2
        elif scale == 'minmax_m1p1':
            trn = 2 * (trn / trn.max(axis=0)) - 1
            val = 2 * (val / val.max(axis=0)) - 1
            tst = 2 * (tst / tst.max(axis=0)) - 1
        elif scale == 'standard':
            scaler = StandardScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn)
            val = scaler.transform(val)
            tst = scaler.transform(tst)
        print(f'{scale} Normalization done')

    # build dataset
    trn_dataset = BuildDataset_default(
            data          = trn, 
            timestamps    = trn_ts, 
            window_size   = window_size, 
            label_len     = label_len,
            pred_len      = pred_len,
            label_data    = label,
            model_type    = model_type)

    val_dataset = BuildDataset_default(
            data          = val, 
            timestamps    = val_ts, 
            window_size   = window_size, 
            label_len     = label_len,
            pred_len      = pred_len,
            label_data    = label,
            model_type    = model_type)

    tst_dataset = BuildDataset_default(
            data          = tst, 
            timestamps    = tst_ts, 
            window_size   = window_size, 
            label_len     = label_len,
            pred_len      = pred_len,
            label_data    = label,
            model_type    = model_type)

    # torch dataloader
    trn_dataloader = DataLoader(trn_dataset,
                                batch_size    =  train_setting.batch_size,
                                shuffle       =  train_setting.shuffle,
                                num_workers   =  train_setting.num_workers,
                                pin_memory    =  train_setting.pin_memory,
                                drop_last     =  True)

    dev_dataloader = DataLoader(val_dataset,
                                batch_size    =  train_setting.batch_size,
                                shuffle       =  train_setting.shuffle,
                                num_workers   =  train_setting.num_workers,
                                pin_memory    =  train_setting.pin_memory,
                                drop_last     =  True)

    tst_dataloader = DataLoader(tst_dataset,
                                batch_size    =  train_setting.test_batch_size,
                                shuffle       =  train_setting.shuffle,
                                num_workers   =  train_setting.num_workers,
                                pin_memory    =  train_setting.pin_memory,
                                drop_last     =  False)

    information_dict = {}
    information_dict.update({'enc_in': next(iter(trn_dataloader))['given']['ts'].size(-1)})
    information_dict.update({'dec_in': next(iter(trn_dataloader))['given']['ts'].size(-1)})
    information_dict.update({'c_out': next(iter(trn_dataloader))['given']['ts'].size(-1)})

    return information_dict, trn_dataloader, dev_dataloader, tst_dataloader