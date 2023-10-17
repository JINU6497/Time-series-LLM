import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.EmbedforTMAE import TMAE_PatchEmbedding
from layers.Conv_Blocks import conv_resize_up_scailing, conv_resizeback_up_scailing
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
import pdb

from einops import rearrange

class TMAE2(nn.Module):
    def __init__(self, configs):
        super(TMAE2, self).__init__()
        self.task_name = configs.taskname
        self.window_size = configs.window_size
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.window_size - self.patch_size) // self.stride + 1
        self.mode = configs.mode
        self.channels = configs.enc_in
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        
        if (self.window_size - self.patch_size) % self.stride != 0:
            padding = self.stride - ((self.window_size - self.patch_size) \
                % self.stride)
            self.patch_num += 1
        else:
            padding = 0
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        
        self.patch_embedding = TMAE_PatchEmbedding(configs.d_model, configs.patch_size, configs.stride, 
                                                    configs.dropout, configs.embed_type, configs.freq)
        
        
    def encoder(self, x, x_mark):
        B, W, D = x.size()
        if self.mode == 'SFT':
            means = x.mean(1, keepdim=True).detach() # B, 1, D
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+1e-5).detach() # B, 1, D
            x = (x - means)/(stdev+0.0001)
        elif self.mode == 'DFT':
            x = self.revin_layer(x, 'norm')
        
        # [batch, feature, window_len + a]
        patch_x = self.padding_patch_layer(x.permute(0, 2, 1))                   

        # [batch, feature, (window_len + a) / patch_len, patch_len ], (window_len + a) / patch_len = (Patch_num)
        patch_x = patch_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        patch_x = rearrange(patch_x, 'batch dimension patch_num patch_size -> (batch dimension) patch_num patch_size')
        
        if x_mark is not None:
            patch_x_mark = self.padding_patch_layer(x_mark.permute(0, 2, 1)) 
            patch_x_mark = patch_x_mark.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            patch_x_mark = rearrange(patch_x_mark, 'batch dimension patch_num patch_size -> (batch dimension) patch_num patch_size')

        x_enc, n_vars = self.patch_embedding(patch_x, patch_x_mark)
        
        outputs = self.gpt2(inputs_embeds=x_enc).last_hidden_state
        
        if self.mode == 'SFT':
            outputs = self.sft_out_layer(outputs.reshape(B*D*self.patch_num, -1))
            outputs = rearrange(outputs, '(batch dimension patch_num) (patch_size) -> batch dimension patch_num patch_size', 
                                batch=B, dimension=D)
        elif self.mode == 'DFT':
            outputs = self.dft_out_layer(outputs.reshape(B*D, -1))
            outputs = rearrange(outputs, '(batch dimension) pred_len -> batch pred_len dimension', 
                                batch=B, dimension=D)
            outputs = self.revin_layer(outputs, 'denorm')
        return patch_x, outputs

    def forecast(self, x_enc, x_mark_enc):
        # Encoder
        return self.encoder(x_enc, x_mark_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, window_size * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc, dec_out = self.forecast(x_enc, x_mark_enc)
            return x_enc, dec_out  
        if self.task_name == 'imputation':
            x_enc, dec_out = self.imputation(x_enc, x_mark_enc)
            return x_enc, dec_out  
        if self.task_name == 'anomaly_detection':
            x_enc, dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return x_enc, dec_out  
        if self.task_name == 'classification':
            x_enc, dec_out = self.classification(x_enc, x_mark_enc)
            return x_enc, dec_out
        return None
    

