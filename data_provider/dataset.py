from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def load_anomaly_dataset(dataname: str, datainfo: dict, subdataname: str = None, valid_split_rate: float = 0.8):
    datainfo = OmegaConf.create(datainfo)
    """
    load dataset
    Parameters
    ----------
    dataname : str
        name of the data
    datainfo : dict
        information of data from config.yaml
    subdataname : str
        name of the sub data (using only for SMD, SMAP and MSL data)
    valid_split_rate : float(default=0.8)
        train, validation split rate
    Returns
    -------
    trainset : ndarray(shape=(time, num of features))
        train dataset
    train_timestamp : ndarray(shape=(time,))
        timestamp of train dataset
    validset : ndarray(shape=(time, num of features))
        validation dataset
    valid_timestamp : ndarray(shape=(time,))
        timestamp of validation dataset
    testset : ndarray(shape=(time, num of features))
        test dataset
    test_timestamp : ndarray(shape=(time,))
        timestamp of test dataset
    test_label : ndarray(shape=(time,))
        attack/anomaly labels of test dataset
    """
    try:
        assert dataname in ['PSM', 'SWaT', 'SMD', 'WADI', 'SMAP', 'MSL', 'HAI', 'Hamon', 'KETI']
    except AssertionError as e:
        raise

    if dataname == 'PSM':
        # trainset = pd.read_csv(datainfo.train_path,
        #                        index_col=0)
        # trainset = trainset.dropna()
        # train_timestamp = trainset.index
        # valid_split_index = int(len(trainset) * valid_split_rate)
        # validset = trainset.iloc[valid_split_index:].to_numpy()
        # trainset = trainset.iloc[:valid_split_index].to_numpy()
        # valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        # train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        # testset = pd.read_csv(datainfo.test_path,
        #                       index_col=0).to_numpy()
        # test_timestamp = np.arange(len(testset))
        # test_label = pd.read_csv(datainfo.test_label_path,
        #                          index_col=0).to_numpy()
        trainset = pd.read_csv(datainfo.train_path,
                               index_col=0).to_numpy()
        trainset = trainset.dropna()
        train_timestamp = trainset.index.to_numpy()
        validset = pd.read_csv(datainfo.valid_path,
                               index_col=0).to_numpy()
        valid_timestamp = validset.index.to_numpy()
        testset = pd.read_csv(datainfo.test_path,
                              index_col=0).to_numpy()
        test_timestamp = np.arange(len(testset))
        test_label = pd.read_csv(datainfo.test_label_path,
                                 index_col=0).to_numpy()

    elif dataname == 'SWaT':
        trainset = pd.read_pickle(datainfo.train_path).drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        testset = pd.read_pickle(datainfo.test_path)
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label = testset['Normal/Attack']
        test_label[test_label == 'Normal'] = 0
        test_label[test_label != 0] = 1
        testset = testset.drop(['Normal/Attack', ' Timestamp'],
                               axis=1).to_numpy()

    elif dataname == 'SMD':
        trainset = np.loadtxt(os.path.join(datainfo.train_dir, f'{subdataname}.txt'),
                              delimiter=',',
                              dtype=np.float32)
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        testset = np.loadtxt(os.path.join(datainfo.test_dir, f'{subdataname}.txt'),
                             delimiter=',',
                             dtype=np.float32)
        test_timestamp = np.arange(len(testset))
        test_label = np.loadtxt(os.path.join(datainfo.test_label_dir, f'{subdataname}.txt'),
                                delimiter=',',
                                dtype=np.int)

    elif dataname == 'WADI':
        trainset = pd.read_csv(datainfo.train_path,
                               index_col=0,
                               header=3).drop(['Date', 'Time'], axis=1)
        columns = trainset.columns
        delete_col = [47, 48, 83, 84]
        trainset = trainset.drop(columns[delete_col], axis=1)
        trainset = trainset.dropna()
        train_timestamp = trainset.index
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset.iloc[valid_split_index:].to_numpy()
        trainset = trainset.iloc[:valid_split_index].to_numpy()
        valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        testset = pd.read_csv(datainfo.test_path,
                              index_col=0).drop(['Date', 'Time'], axis=1)
        testset = testset.dropna(axis=1).to_numpy()
        test_timestamp = np.arange(len(testset))
        test_label = pd.read_csv(datainfo.test_label_path,
                                 index_col=0).to_numpy()

    elif dataname == 'HAI':
        trainset = sorted([x for x in Path(datainfo.data_dir).glob("train*.csv")])
        testset = sorted([x for x in Path(datainfo.data_dir).glob("test*.csv")])
        trainset = dataframe_from_csvs(trainset)
        testset = dataframe_from_csvs(testset)
        TIMESTAMP_FIELD = "timestamp"
        ATTACK_FIELD = "Attack"
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset.iloc[valid_split_index:]
        trainset = trainset.iloc[:valid_split_index]
        train_timestamp = trainset[TIMESTAMP_FIELD].to_numpy()
        trainset = trainset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()
        valid_timestamp = validset[TIMESTAMP_FIELD].to_numpy()
        validset = validset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()
        test_timestamp = testset[TIMESTAMP_FIELD].to_numpy()
        test_label = testset[ATTACK_FIELD].to_numpy()
        testset = testset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()


    elif dataname == 'Hamon':
        # trainset = pd.read_csv(os.path.join(datainfo.data_dir, f'Train_{subdataname}.csv'), index_col=0).dropna()
        # train_timestamp = trainset.index
        # valid_split_index = int(len(trainset) * valid_split_rate)
        # validset = trainset.iloc[valid_split_index:].to_numpy()
        # trainset = trainset.iloc[:valid_split_index].to_numpy()
        # valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        # train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        # testset = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_{subdataname}.csv'), index_col=0).dropna()
        # test_timestamp = testset.index
        # test_label = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_Label_{subdataname}.csv'), index_col=0).to_numpy()

        trainset = pd.read_csv(os.path.join(datainfo.data_dir, f'Train_{subdataname}.csv')).dropna()
        validset = pd.read_csv(os.path.join(datainfo.data_dir, f'Valid_{subdataname}.csv')).dropna()
        testset = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_{subdataname}.csv')).dropna()
        if subdataname == 'new2':
            trainset = trainset.drop(['mng_no', 'if_idx', 'svr_no'], axis=1)
            validset = validset.drop(['mng_no', 'if_idx', 'svr_no'], axis=1)
            testset = testset.drop(['mng_no', 'if_idx', 'svr_no'], axis=1)
        # print(trainset.shape, validset.shape, testset.shape)
        # print(trainset.columns)
        TIMESTAMP_FIELD = "ymdhms"
        ATTACK_FIELD = "target"
        train_timestamp = trainset[TIMESTAMP_FIELD].astype(str)
        valid_timestamp = validset[TIMESTAMP_FIELD].astype(str)
        # train_timestamp = pd.to_datetime(train_timestamp, format='%Y%m%d%H%M%S%f')
        # valid_timestamp = pd.to_datetime(valid_timestamp, format='%Y%m%d%H%M%S%f')
        trainset = trainset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()
        validset = validset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()
        test_timestamp = testset[TIMESTAMP_FIELD].astype(str)
        # test_timestamp = pd.to_datetime(test_timestamp, format='%Y%m%d%H%M%S%f')
        test_label = testset[ATTACK_FIELD].to_numpy()
        testset = testset.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1).to_numpy()

    elif dataname == 'KETI':
        trainset = pd.read_csv(os.path.join(datainfo.data_dir, f'Train_{subdataname}.csv'), index_col=0).dropna()
        trainset = trainset.reset_index(drop=True)
        train_timestamp = trainset.index
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset.iloc[valid_split_index:].to_numpy()
        trainset = trainset.iloc[:valid_split_index].to_numpy()
        valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        testset = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_{subdataname}.csv'), index_col=0).dropna()
        testset = testset.reset_index(drop=True)
        test_timestamp = testset.index
        test_label = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_Label_{subdataname}.csv'), index_col=0).to_numpy()
        
    else:
        trainset = np.load(os.path.join(datainfo.train_dir, f'{subdataname}.npy'))
        testset = np.load(os.path.join(datainfo.test_dir, f'{subdataname}.npy'))
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label_info = pd.read_csv(datainfo.test_label_path, index_col=0).loc[subdataname]
        test_label = np.zeros([int(test_label_info.num_values)], dtype=np.int)

        for i in eval(test_label_info.anomaly_sequences):
            if type(i) == list:
                test_label[i[0]:i[1] + 1] = 1
            else:
                test_label[i] = 1

    return trainset, train_timestamp, validset, valid_timestamp, testset, test_timestamp, test_label


def load_custom_dataset(dataname: str, datainfo: dict, window_size: int, label_len: int, pred_len:int, 
                        split_rate: list = [0.7, 0.1, 0.2], timeenc = 0, freq='h'):
    datainfo = OmegaConf.create(datainfo)    
    
    try:
        assert dataname in ['custom', 'etth', 'ettm']
    except AssertionError as e:
        raise
    
    if sum(split_rate) != 1.0:
        raise ValueError("The total sum of split_rate must be 1.0.")
    
    if dataname == 'custom':
        df_raw = pd.read_csv(datainfo.datadir)
        cols = list(df_raw.columns)
        cols.remove('date')
        num_train, num_valid, num_test  = int(len(df_raw) * split_rate[0]), int(len(df_raw) * split_rate[1]), int(len(df_raw) * split_rate[2])
        df_data = df_raw[cols]
        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date']).copy()
        if timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).to_numpy()
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].to_numpy()), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        df_data, df_stamp = df_data.to_numpy(), df_stamp.to_numpy()
        trainset, train_timeembedding = df_data[:num_train], data_stamp[:num_train]
        validset, valid_timeembedding = df_data[num_train:num_train + num_valid], data_stamp[num_train:num_train + num_valid]
        testset, test_timeembedding = df_data[num_train + num_valid:num_train + num_valid + num_test], data_stamp[num_train + num_valid:num_train + num_valid + num_test]
        
        if dataname == 'etth1' or dataname == 'etth2':
            df_raw = pd.read_csv(datainfo.datadir)
            cols = list(df_raw.columns)
            cols.remove('date')
            df_data = df_raw[cols]
            df_stamp = df_raw[['date']].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date']).copy()
            if timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis = 1).to_numpy()
            elif timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].to_numpy()), freq=freq)
                data_stamp = data_stamp.transpose(1, 0)
                
            df_data, df_stamp = df_data.to_numpy(), df_stamp.to_numpy()
            trainset, train_timeembedding = df_data[ : 12 * 30 * 24], data_stamp[ : 12 * 30 * 24]
            validset, valid_timeembedding = df_data[12 * 30 * 24 - window_size : 12 * 30 * 24 + 4 * 30 * 24], data_stamp[12 * 30 * 24 - window_size : 12 * 30 * 24 + 4 * 30 * 24]
            testset, test_timeembedding = df_data[12 * 30 * 24 + 4 * 30 * 24 - window_size : 12 * 30 * 24 + 8 * 30 * 24], data_stamp[12 * 30 * 24 + 4 * 30 * 24 - window_size : 12 * 30 * 24 + 8 * 30 * 24]

        if dataname == 'ettm1' or dataname == 'ettm2':
            df_raw = pd.read_csv(datainfo.datadir)
            cols = list(df_raw.columns)
            cols.remove('date')
            df_data = df_raw[cols]
            df_stamp = df_raw[['date']].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date']).copy()
            if timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], axis = 1).to_numpy()
            elif timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].to_numpy()), freq=freq)
                data_stamp = data_stamp.transpose(1, 0)
                
            df_data, df_stamp = df_data.to_numpy(), df_stamp.to_numpy()
            trainset, train_timeembedding = df_data[ : 12 * 30 * 24 * 4], data_stamp[ : 12 * 30 * 24 * 4]
            validset, valid_timeembedding = df_data[12 * 30 * 24 * 4 - window_size : 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4], data_stamp[12 * 30 * 24 * 4 - window_size : 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
            testset, test_timeembedding = df_data[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - window_size : 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4], data_stamp[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - window_size : 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        
    return trainset, train_timeembedding, validset, valid_timeembedding, testset, test_timeembedding

