import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_Encoder(nn.Module):
    def __init__(self, d_model):
        
        self.enc1 = nn.Conv2d(
            in_channels=d_model, out_channels=d_model//2, kernel_size=3, 
            stride=1, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=d_model//2, out_channels=(d_model//2)//2, kernel_size=3, 
            stride=1, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=(d_model//2)//2, out_channels=((d_model//2)//2)//2, kernel_size=3, 
            stride=1, padding=1
        )
        self.relu = F.relu

    def forward(self, input):
        x = self.relu(self.enc1(input))
        x = self.relu(self.enc2(x))
        output = self.relu(self.enc3(x))
        return output
    
class Lambda(nn.Module):
    def __init__(self, d_model, latent_dim = 128, training = True):
        super(Lambda, self).__init__()
        
        self.training = training
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.hidden_to_mean = nn.Linear(((self.d_model//2)//2)//2, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(((self.d_model//2)//2)//2, self.latent_dim)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, x_encoded):
        self.latent_mean = self.hidden_to_mean(x_encoded)
        self.latent_logvar = self.hidden_to_logvar(x_encoded)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean
        
class VAE_Decoder(nn.Module):
    def __init__(self, latent_length, d_model):
        self.latent_length = latent_length
        self.latent_to_d_model=nn.Linear(self.latent_length, ((d_model//2)//2)//2)
        
        self.enc1 = nn.ConvTranspose2d(
            in_channels=((d_model//2)//2)//2, out_channels=((d_model//2)//2), kernel_size=3, 
            stride=1, padding=1
        )
        self.enc2 = nn.ConvTranspose2d(
            in_channels=((d_model//2)//2), out_channels=d_model//2, kernel_size=3, 
            stride=1, padding=1
        )
        self.enc3 = nn.ConvTranspose2d(
            in_channels=d_model//2, out_channels=d_model, kernel_size=3, 
            stride=1, padding=1
        )
        self.relu = F.relu

    def forward(self, input):
        h_state = self.latent_to_d_model(input)
        x = self.relu(self.enc1(h_state))
        x = self.relu(self.enc2(x))
        output = self.relu(self.enc3(x))
        return output