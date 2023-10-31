import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.EmbedforTMAE import TMAE_Embedding, TMAE_patching
from layers.Conv_Blocks import MultiScaleAugmentation, MultiScaleAugmentationBack
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
from timm.models.vision_transformer import PatchEmbed, Block
from layers.RevIN import RevIN
from einops import rearrange

import pdb


def FFT_for_Period(x, k=10):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    freq = torch.median(top_list.detach().cpu())
    period = x.shape[1] // freq
    return period

class TMAE2(nn.Module):
    def __init__(self, configs):
        super(TMAE2, self).__init__()
        self.configs = configs
        self.task_name = configs.taskname
        self.window_size = configs.window_size
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.window_size - self.patch_size) // self.stride + 1
        self.mode = configs.mode
        self.channels = configs.enc_in
        self.d_ff = configs.d_ff
        self.enc_d_model = configs.enc_d_model
        self.mask_ratio = configs.mask_ratio
        
        if (self.window_size - self.patch_size) % self.stride != 0:
            padding = self.stride - ((self.window_size - self.patch_size) \
                % self.stride)
            self.patch_num += 1
        else:
            padding = 0
            
        self.padding_layer = nn.ReplicationPad1d((0, padding))
        
        self.embedding = TMAE_Embedding(channels     = configs.enc_in, 
                                        d_model      = configs.enc_d_model, 
                                        dropout      = configs.dropout, 
                                        embed_type   = configs.embed_type, 
                                        freq         = configs.freq)
        
        self.patching = TMAE_patching(d_model  = configs.enc_d_model, 
                                      patch_size   = self.patch_size, 
                                      dropout      = configs.dropout)

        self.multi_scale_augmentation = MultiScaleAugmentation(d_model = configs.enc_d_model,
                                                               num_kernels = 2, 
                                                               init_weight = True)
        
        self.multi_scale_augmentation_back = MultiScaleAugmentationBack(d_model = configs.enc_d_model,
                                                                num_kernels = 2, 
                                                                init_weight = True)
        self.revin_layer = RevIN(self.channels)
        # --------------------------------------------------------------------------
        # Encoder specifics

        self.pos_embed = nn.Parameter(torch.zeros(1, 300, configs.enc_d_model), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(configs.enc_d_model, configs.enc_n_heads, configs.mlp_ratio, 
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            
            for i in range(configs.e_layers)])
        self.norm = nn.LayerNorm(configs.enc_d_model)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed = nn.Linear(configs.enc_d_model, configs.dec_d_model, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, configs.dec_d_model))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 300, configs.dec_d_model), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(configs.dec_d_model, configs.dec_n_heads, configs.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(configs.d_layers)])

        self.decoder_norm = nn.LayerNorm(configs.dec_d_model)
        self.decoder_pred = nn.Linear(configs.dec_d_model, configs.patch_size**2 * self.enc_d_model, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

    def forward_preprocessing(self, x, x_mark):
        period = FFT_for_Period(x)
        B, T, N = x.size()
        x = self.revin_layer(x, 'norm')
        
        if self.window_size % period != 0:
            length = ((self.window_size // period) + 1) * period
            # padding = torch.zeros([x.shape[0], (length - self.window_size), x.shape[2]]).to(x.device)
            padding_patch_layer = nn.ReplicationPad1d((0, length - self.window_size))
            out = padding_patch_layer(x.permute(0,2,1))
        else:
            length = self.window_size
            out = x.permute(0,2,1)
            
        # reshape
        out = rearrange(out, 'batch dimension ( frequency period ) -> batch dimension frequency period',
                        period = period)
        
        if x_mark is not None:
            x_mark_out = padding_patch_layer(x_mark.permute(0,2,1))
            x_mark_out = rearrange(x_mark_out, 'batch dimension ( frequency period ) -> batch dimension frequency period',
                               period = period)
            
        # embedding        
        x_enc, n_vars = self.embedding(out, x_mark_out)

        padded_input, hw_list = self.padding_for_patching(x_enc, self.patch_size)
        patched_input = self.patching(padded_input)
        
        mask_list=[]
        latent_list=[]
        pred_list=[]
        
        for i in range(len(patched_input)):
            latent, mask, ids_restore = self.forward_encoder(patched_input[i], self.mask_ratio) # [B, patch_num, d_model]
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            latent_list.append(latent)
            mask_list.append(mask)
            pred_list.append(self.unpatchify(pred, hw_list[i]))
            
        output_list = self.multi_scale_augmentation_back(pred_list)
        # print(output_list[0].shape, output_list[1].shape, output_list[2].shape, output_list[3].shape, output_list[4].shape)
        # print(padded_input[0].shape, padded_input[1].shape, padded_input[2].shape, padded_input[3].shape, padded_input[4].shape)
        breakpoint()

        final_output = sum(output_list)/len(output_list)
        breakpoint()

        # outputs = rearrange(outputs, '(batch dimension) pred_len -> batch pred_len dimension', 
        #                     batch=B, dimension=D)
        outputs = self.revin_layer(outputs, 'denorm')
        return x
    
    def unpatchify(self, x, hw):
        """
        x: (N, L, patch_size**2 * channel)
        imgs: (N, channel, H, W)
        """
        p = self.patch_size
        dimension=self.enc_d_model
        h, w = int(hw[0]/p), int(hw[1]/p)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, dimension))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], dimension, h * p, w * p))
        return imgs
    
    def padding_for_patching(self, input, patch_size):
        out_list = []
        hw_list = []
        B, N, H, W = input.size()
        length_W = (patch_size - (W % patch_size)) % patch_size
        length_H = (patch_size - (H % patch_size)) % patch_size
        padding_for_patch = nn.ReplicationPad2d((0, length_W, 0, length_H))
        out = padding_for_patch(input)
        out_list.append(out)
        hw_list.append((H+length_H,W+length_W))
        return out_list, hw_list
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # add patch pos embed
        x = x + self.pos_embed[:, :x.size(1), :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # add pos embed
        x = x + self.decoder_pos_embed[:,x.size(1),:]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x
    
    def forecast(self, x_enc, x_mark_enc):
        # Encoder
        return self.forward_preprocessing(x_enc, x_mark_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.forward_preprocessing(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.forward_preprocessing(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.forward_preprocessing(x_enc)
        # Output
        # (batch_size, window_size * enc_d_model)
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
    

