import torch
import torch.nn as nn
import torch.nn.functional as F


class MySample():
    def __init__(self, generator: nn.Module):
        self.generator = generator
        self.generator.eval()
        self.dim_latent = 100

        
    def sample(self, num_sample: int=10):
        dim_channel = 1
        dim_height  = 32
        dim_width   = 32
        sample = torch.zeros(num_sample, dim_channel, dim_height, dim_width)

        with torch.no_grad():
            # generate samples
            z = torch.randn(num_sample, self.dim_latent, 1, 1)
            fake = self.generator(z)
            sample.copy_(fake)

        return sample


