import torch
import torch.nn as nn
import torch.nn.functional as F
from MyModel import MyVariationalEncoder

class MySample():
    def __init__(self, vae: MyVariationalEncoder):
        self.latern     = vae.latent
        self.decoder    = vae.decoder
        self.decoder.eval()
        
        self.dim_latent     = vae.dim_latent
        self.dim_channel    = vae.dim_channel
        self.dim_height     = vae.dim_height
        self.dim_width      = vae.dim_width

 
    def sample(self, num_sample: int=10):

        with torch.no_grad():
            
            z = torch.randn(num_sample, self.dim_latent, 1, 1)
            sample = self.decoder(z)

        return sample


