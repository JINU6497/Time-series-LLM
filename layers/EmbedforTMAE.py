import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from einops import rearrange
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
    def __init__(self, c_in, d_model):
        super(TMAE_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x)
        return x


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

class TMAE_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_position=512):
        super(TMAE_PositionalEmbedding, self).__init__()
        self.position_lookuptable = nn.Embedding(max_position, 1)
        torch.nn.init.xavier_uniform_(self.position_lookuptable.weight.data)
    def forward(self, x):
        positional_encoding = self.position_lookuptable.weight[:x.size(-2)*x.size(-1)]
        positional_encoding = positional_encoding.reshape(1, 1, x.size(-2), x.size(-1))
        return positional_encoding.expand(x.size(0), 1, x.size(-2), x.size(-1))
    
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
        return (hour_x + weekday_x + day_x + month_x + minute_x).permute(0,3,1,2).contiguous()
    
class TMAE_Embedding(nn.Module):
    def __init__(self, channels, d_model, dropout, embed_type, freq):
        super(TMAE_Embedding, self).__init__()        
        # Token Embedding
        self.TokenEmbedding = TMAE_TokenEmbedding(channels, d_model)
        # Positional embedding
        self.position_embedding = TMAE_PositionalEmbedding(d_model)
        # Temporal Embedding
        self.temporal_embedding = TMAE_TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        # [batch, faeture, frequency, period]
        B, N, H, W = x.size()
        
        TokenEmbedding = self.TokenEmbedding(x) # [batch, d_model, frequency, period]
        PositionalEncoding = self.position_embedding(x)  # [batch, 1, frequency, period]
        
        if x_mark is None:
            output = TokenEmbedding + PositionalEncoding
        else:
            Temporal_embedding = self.temporal_embedding(x_mark)
            output = TokenEmbedding + PositionalEncoding + Temporal_embedding
        return self.dropout(output), N

class TMAE_patching(nn.Module):
    def __init__(self, d_model, patch_size, dropout):
        super(TMAE_patching, self).__init__()        
        self.TokenEmbedding = nn.Conv2d(d_model, d_model, patch_size, patch_size)
    def forward(self, input):
        input = self.TokenEmbedding(input)
        input = rearrange(input, 'B D H W -> B (H W) D')
        return input
    
class TMAE_patching2(nn.Module):
    def __init__(self, d_model, patch_size, dropout):
        super(TMAE_patching, self).__init__()        
        self.TokenEmbedding = nn.Conv2d(d_model, d_model, patch_size, patch_size)
    def forward(self, input_list):
        output_list = []
        for i in input_list:
            input = self.TokenEmbedding(i)
            input = rearrange(input, 'B D H W -> B (H W) D')
            output_list.append(input)
        return output_list