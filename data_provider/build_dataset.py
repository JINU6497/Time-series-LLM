from torch.utils.data import Dataset
import numpy as np

class BuildDataset_default(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 label_len: int,
                 pred_len: int,
                 label_data: np.ndarray = None,
                 model_type: str = 'forecasting'):
        
        self.data = data
        self.timestamps = timestamps
        self.window_size = window_size
        self.model_type = model_type
        self.label_data = label_data
        self.label_len = label_len
        self.pred_len = pred_len
        
        if model_type == 'reconstruction':
            self.valid_window = len(self.data) - self.window_size + 1
        elif model_type == 'forecasting':
            self.valid_window = len(self.data) - self.window_size - self.pred_len + 1
        print(f"# of valid windows: {self.valid_window}")

    def __getitem__(self, idx):
        if self.model_type == 'reconstruction':
            start = idx
            end = start + self.window_size
            item = { 'given':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]},
                    'target':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]}
                    }
            if self.label_data:
                item['target'].update({'anomaly_label': self.label_data[start:end]})
            
        elif self.model_type == 'forecasting':
            g_start = idx
            g_end = g_start + self.window_size
            t_start = g_end - self.label_len
            t_end = t_start + self.label_len + self.pred_len
            item = { 'given':{
                        'ts': self.data[g_start:g_end],
                        'timeembedding': self.timestamps[g_start:g_end]},
                    'target':{
                        'ts': self.data[t_start:t_end],
                        'timeembedding': self.timestamps[t_start:t_end]}
                    }
            if self.label_data:
                item['target'].update({'anomaly_label': self.label_data[t_start:t_end]})
                
        return item

    def __len__(self):        
        return self.valid_window

class BuildDataset_pretraining(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 model_type: str = 'reconstruction'):
        
        self.data = data
        self.timestamps = timestamps
        self.window_size = window_size
        self.model_type = model_type
        self.valid_window = len(self.data) - self.window_size + 1
        print(f"# of valid windows: {self.valid_window}")

    def __getitem__(self, idx):
        if self.model_type == 'reconstruction':
            start = idx
            end = start + self.window_size
            item = { 'given':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]},
                    'target':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]}
                    }
        return item

    def __len__(self):        
        return self.valid_window
    
class BuildDataset_finetuning(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 label_len: int,
                 pred_len: int,
                 label_data: np.ndarray = None,
                 model_type: str = 'forecasting'):
        
        self.data = data
        self.timestamps = timestamps
        self.window_size = window_size
        self.model_type = model_type
        self.label_data = label_data
        self.label_len = label_len
        self.pred_len = pred_len
        
        if model_type == 'reconstruction':
            self.valid_window = len(self.data) - self.window_size + 1
        elif model_type == 'forecasting':
            self.valid_window = len(self.data) - self.window_size - self.pred_len + 1
        print(f"# of valid windows: {self.valid_window}")

    def __getitem__(self, idx):
        if self.model_type == 'reconstruction':
            start = idx
            end = start + self.window_size
            item = { 'given':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]},
                    'target':{
                        'ts': self.data[start:end],
                        'timeembedding': self.timestamps[start:end]}
                    }
            if self.label_data:
                item['target'].update({'anomaly_label': self.label_data[start:end]})
            
        elif self.model_type == 'forecasting':
            g_start = idx
            g_end = g_start + self.window_size
            t_start = g_end - self.label_len
            t_end = t_start + self.label_len + self.pred_len
            item = { 'given':{
                        'ts': self.data[g_start:g_end],
                        'timeembedding': self.timestamps[g_start:g_end]},
                    'target':{
                        'ts': self.data[t_start:t_end],
                        'timeembedding': self.timestamps[t_start:t_end]}
                    }
            if self.label_data:
                item['target'].update({'anomaly_label': self.label_data[t_start:t_end]})
                
        return item

    def __len__(self):        
        return self.valid_window