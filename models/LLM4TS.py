import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import LLM4TS_PatchEmbedding
from layers.RevIN import RevIN
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class LLM4TS(nn.Module):
    def __init__(self, configs):
        super(LLM4TS, self).__init__()
        self.task_name = configs.taskname
        self.window_size = configs.window_size
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.window_size - self.patch_size) // self.stride + 1
        self.channel_independence = configs.channel_independence
        self.channels = configs.enc_in
        self.mode = configs.mode
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        
        if (self.window_size - self.patch_size) % self.stride != 0:
            padding = self.stride - ((self.window_size - self.patch_size) \
                % self.stride)
            self.patch_num += 1
        else:
            padding = 0
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        
        self.patch_embedding = LLM4TS_PatchEmbedding(configs.d_model, configs.patch_size, configs.stride, 
                                                    configs.dropout, configs.embed_type, configs.freq)

        if self.mode == 'SFT':
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        
            train_params = ['ln', 'wpe', 'wte']
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if any(x in name for x in train_params):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            self.sft_out_layer = nn.Linear(configs.d_model, configs.patch_size)

        elif self.mode == 'DFT':
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                param.requires_grad = False
                
            self.revin_layer = RevIN(self.channels)
            for i, (name, param) in enumerate(self.revin_layer.named_parameters()):
                param.requires_grad = False

            self.dft_out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
            
        # Lora
        lora_target_modules = ['c_attn', 'c_proj']
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            fan_in_fan_out=True,
        )
        
        self.gpt2 = get_peft_model(self.gpt2, lora_config)
        self.gpt2.print_trainable_parameters()
        
    def encoder(self, x, x_mark):
        B, W, D = x.size()
        if self.mode == 'SFT':
            means = x.mean(1, keepdim=True).detach() # B, 1, D
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+1e-5).detach() # B, 1, D
            breakpoint()
            x = (x - means)/(stdev+0.0001)
        elif self.mode == 'DFT':
            x = self.revin_layer(x, 'norm')
        
        # [batch, feature, window_len + a]
        patch_x = self.padding_patch_layer(x.permute(0, 2, 1))                   

        # [batch, feature, (window_len + a) / patch_len, patch_len ], (window_len + a) / patch_len = (Patch_num)
        patch_x = patch_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        if x_mark is not None:
            patch_x_mark = self.padding_patch_layer(x_mark.permute(0, 2, 1)) 
            patch_x_mark = patch_x_mark.unfold(dimension=-1, size=self.patch_size, step=self.stride)

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
    

