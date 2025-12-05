import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# define your own network architecture for Generator
# 
# DIMENSION NOTE:
# [input]   : dimension of input to the generator is (N, dim_latent, 1, 1)
# [output]  : dimension of output from the generator is (N, 1, 32, 32)
# =========================================================================
class MyGenerator(nn.Module):
    def __init__(self, dim_channel: int=1, dim_height: int=32, dim_width: int=32):
        super(MyGenerator, self).__init__()
        self.dim_channel    = dim_channel
        self.dim_height     = dim_height
        self.dim_width      = dim_width
        self.dim_latent = 100
        # define a generative network 
        self.pipe =  nn.Sequential(
            nn.ConvTranspose2d(self.dim_latent, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, self.dim_channel, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        #x = torch.randn(x.size(0), self.dim_channel, self.dim_height, self.dim_width)  # dummy implementation
        x = self.pipe(x)
        return x

# =========================================================================
# define your own network architecture for Discriminator
#
# DIMENSION NOTE:
# [input]   : dimension of input to the discriminator is (N, 1, 32, 32)
# [output]  : dimension of output from the discriminator is (N, 1, 1, 1)
# =========================================================================
class MyDiscriminator(nn.Module):
    def __init__(self, dim_channel: int=1, dim_height: int=32, dim_width: int=32):
        super(MyDiscriminator, self).__init__()
        self.dim_channel    = dim_channel
        self.dim_height     = dim_height
        self.dim_width      = dim_width
        # define a discriminative network 

        self.pipe = nn.Sequential(
            nn.Conv2d(self.dim_channel, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = torch.randn(x.size(0), 1, 1, 1)  # dummy implementation
        x = self.pipe(x)
        return x
