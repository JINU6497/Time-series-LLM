import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C] [batch, Timedata length, input dimnesion]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.window_size = configs.window_size
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        # d_model = in_channels = 16
        # d_ff = out_channels = 32
        # num_kernels = 6
        # kernel_size = 2*i + 1, padding = i
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # batch, window, variable
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.window_size + self.pred_len) % period != 0:
                length = (
                    ((self.window_size + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.window_size + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.window_size + self.pred_len)
                out = x
            # reshape
            # torch.Size([32, 16, 48, 4])
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # torch.Size([32, 32, 48, 4])
            # res_list[0].shape -> torch.Size([32, 32, 48, 4])
            # res_list[1].shape -> torch.Size([32, 32, 48, 4])
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.window_size + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        # window_size
        self.window_size = configs.window_size
        self.label_len = configs.label_len
        # Pred_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                   for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        self.predict_linear = nn.Linear(
            self.window_size, self.pred_len + self.window_size)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        
        """
        
        DataEmbedding 
        x_mark는 window_size 내에서  label_len 만큼의 예측에 영향을 줄 어느 정도의 길이.
        
        x_mark 있는 경우 -> x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        x_mark 없는 경우 -> x = self.value_embedding(x) + self.position_embedding(x)
        
        """
        
        # batch, pred_len,  
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]