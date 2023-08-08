import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import conv_resize_up_scailing, conv_resizeback_up_scailing
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
import pdb

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesMaskingBlock(nn.Module):
    def __init__(self, configs):
        super(TimesMaskingBlock, self).__init__()
        self.window_size = configs.window_size
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        self.conv_resize_up_scailing = conv_resize_up_scailing(in_channels = configs.d_model, out_channels = configs.d_model)
        self.conv_resizeback_up_scailing = conv_resizeback_up_scailing(in_channels = configs.d_model, out_channels = configs.d_model)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # self.decoder = Decoder(
        # [
        #     DecoderLayer(
        #         AttentionLayer(
        #             FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #             configs.d_model, configs.n_heads),
        #         AttentionLayer(
        #             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #             configs.d_model, configs.n_heads),
        #         configs.d_model,
        #         configs.d_ff,
        #         dropout=configs.dropout,
        #         activation=configs.activation,
        #     )
        #     for l in range(configs.d_layers)
        # ],
        # norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        conv_masked_output = []
        total_input = []
        total_masked_output = []
        
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.window_size + self.pred_len) % period != 0:
                length = (((self.window_size + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.window_size + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.window_size + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv: from 1d Variation to 2d Variation
            conv_output_list = self.conv_resize_up_scailing(out)

            for i in range(len(conv_output_list)):
                Bm, Dm, Hm, Wm = conv_output_list[i].size()
                masking_input = self.mask_specific_size(input_tensor = conv_output_list[i], 
                                                            mask_size = (3, 3),
                                                            num_masked = 3 * (i+1)).reshape(Bm, -1, Dm)
                conv_masked_output_flatten = (self.encoder(masking_input, attn_mask=None))
                conv_masked_output.append(conv_masked_output_flatten.reshape(Bm, Dm, Hm, Wm))

                total_input.append(conv_output_list[i].reshape(Bm, -1, Dm).detach().cpu().numpy())
                total_masked_output.append(conv_masked_output_flatten.detach().cpu().numpy())
            
            out = self.conv_resizeback_up_scailing(conv_masked_output)
            conv_masked_output = []
            
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
    
    def mask_specific_size(self, input_tensor, mask_size = (3,3), num_masked = 5):
        batch_size, num_channels, height, width = input_tensor.size()
        masksize_H, masksize_W = mask_size
        if height > masksize_H or width > masksize_W:
            return input_tensor 

        num_regions_H = height - masksize_H + 1
        num_regions_W = width - masksize_W + 1
        total_regions = num_regions_H * num_regions_W

        num_masked_regions = num_masked

        random_indices = torch.randperm(total_regions)[:num_masked_regions]

        start_h = random_indices // num_regions_W
        start_w = random_indices % num_regions_W

        mask = torch.ones_like(input_tensor)

        for i in range(num_masked_regions):
            h = start_h[i]
            w = start_w[i]
            mask[:, :, h:h + masksize_H, w:w + masksize_W] = 0
            
        x_masked = input_tensor * mask
        return x_masked


class TMAE1(nn.Module):
    def __init__(self, configs):
        super(TMAE1, self).__init__()
        self.configs = configs
        self.task_name = configs.taskname
        self.window_size = configs.window_size
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.pretrain = configs.pretrain

        self.model = nn.ModuleList([TimesMaskingBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        
        if self.pretrain:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        else:
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                self.predict_linear = nn.Linear(
                    self.window_size, self.pred_len + self.window_size)
                self.projection = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
            if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                self.projection = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
            if self.task_name == 'classification':
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(
                    configs.d_model * configs.window_size, configs.num_class)
    
    def pretraining(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        return dec_out
    

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.pretrain:
            dec_out = self.pretraining(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            if self.task_name == 'imputation':
                dec_out = self.imputation(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                return dec_out  # [B, L, D]
            if self.task_name == 'anomaly_detection':
                dec_out = self.anomaly_detection(x_enc)
                return dec_out  # [B, L, D]
            if self.task_name == 'classification':
                dec_out = self.classification(x_enc, x_mark_enc)
                return dec_out  # [B, N]
        return None