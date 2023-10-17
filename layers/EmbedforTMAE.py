import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
import math

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TMAE_TokenEmbedding(nn.Module):
    def __init__(self, batch_size, c_in):
        super(TMAE_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv2d(in_channels=batch_size*c_in, out_channels=batch_size*c_in,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class TMAE_TokenEmbedding(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, 
                 kernel_adjust = 2, num_kernels = 3, init_weight=True):
        super(TMAE_TokenEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(1, self.num_kernels + 1):
            kernels.append(nn.ConvTranspose2d(in_channels=batch_size*in_channels, 
                                              out_channels=batch_size*in_channels, 
                                              kernel_size= 2 * i - 1, 
                                              stride = i, 
                                              padding = (2 * i - 1) // 2))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        return res_list


class TMAE_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, max_len=5000, max_patches=100):
        super(TMAE_PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TMAE_PatchPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_patches=50):
        super(TMAE_PatchPositionalEmbedding, self).__init__()
        self.patch_lookuptable = nn.Embedding(max_patches, 1)
        torch.nn.init.xavier_uniform_(self.patch_lookuptable.weight.data)
        
    def forward(self, x):
        patch_indices = torch.arange(x.size(1)).long().unsqueeze(0).to(x.device)
        return self.patch_lookuptable(patch_indices)
    
class TMAE_TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TMAE_TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,4,:,:]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,3,:,:])
        weekday_x = self.weekday_embed(x[:,2,:,:])
        day_x = self.day_embed(x[:,1,:,:])
        month_x = self.month_embed(x[:,0,:,:])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class TMAE_PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, embed_type, freq):
        super(TMAE_PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Token Embedding
        self.TokenEmbedding = TMAE_TokenEmbedding(patch_len, d_model)
        # Positional embedding
        self.patch_position_embedding = TMAE_PatchPositionalEmbedding(d_model)
        # Temporal Embedding
        self.temporal_embedding = TMAE_TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):

        first_x_mark = x_mark[:,:,:,0].unsqueeze(-1) # [batch, feature, Patch_num , 1]
        
        B, N, patch_num, patch_size = x.size()
        
        # [batch * faeture, Patch_num , patch_len ]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        TokenEmbedding = self.TokenEmbedding(x) # [batch * faeture, Patch_num, d_model]
        PositionalEncoding = self.patch_position_embedding(x)  # [1, Patch_num, 1]
        
        if x_mark is None:
            output = TokenEmbedding + PositionalEncoding
        else:
            Temporal_embedding = self.temporal_embedding(first_x_mark)
            output = TokenEmbedding.reshape(B, N, patch_num, self.d_model) + Temporal_embedding.permute(0,2,1,3)
            output = output.reshape(B*N, patch_num, self.d_model) + PositionalEncoding
            
        return self.dropout(output), N
