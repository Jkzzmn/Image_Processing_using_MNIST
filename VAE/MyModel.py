import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# definition of Variational Encoder
# 
# DIMENSION NOTE:
# [input]   : dimension of input is (N, dim_channel, dim_height, dim_width)
# [output]  : dimension of output is (N, dim_channel, dim_height, dim_width)
# =========================================================================
class MyVariationalEncoder(nn.Module):
    def __init__(self):
        super(MyVariationalEncoder, self).__init__()
        self.dim_latent     = 20   # dummy implementation
        self.dim_channel    = 1
        self.dim_height     = 32
        self.dim_width      = 32 
        self.encoder        = MyEncoder(self.dim_latent, self.dim_channel, self.dim_height, self.dim_width)
        self.latent         = MyLatent(self.dim_latent)
        self.decoder        = MyDecoder(self.dim_latent, self.dim_channel, self.dim_height, self.dim_width)


    def forward(self, x):
        # define forward propagation
        mu, logvar = self.encoder(x)
        z = self.latent(mu, logvar)
        x_recons = self.decoder(z)
        return x_recons, mu, logvar


# =========================================================================
# definition of Encoder
# 
# DIMENSION NOTE:
# [input]   : dimension of input is (N, dim_channel, dim_height, dim_width)
# [output]  : dimension of output is ((N, dim_latent, 1, 1), (N. dim_latent, 1, 1))
# =========================================================================
class MyEncoder(nn.Module):
    def __init__(self, dim_latent: int=100, dim_channel: int=1, dim_height: int=32, dim_width: int=32):
        super(MyEncoder, self).__init__()
        self.dim_latent     = dim_latent
        self.dim_channel    = dim_channel
        self.dim_height     = dim_height
        self.dim_width      = dim_width
        # define a network 

        self.pipe = nn.Sequential(
            nn.Conv2d(dim_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, dim_channel, dim_height, dim_width)
            dummy_out = self.pipe(dummy)           
            self._conv_out_shape = dummy_out.shape  
            total_feature = dummy_out.numel()    

        self.fc_mu = nn.Linear(total_feature,dim_latent)
        self.fc_logvar = nn.Linear(total_feature,dim_latent)



    def forward(self, x):
        x = self.pipe(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        mu     = mu.unsqueeze(-1).unsqueeze(-1)
        logvar = logvar.unsqueeze(-1).unsqueeze(-1)

        return mu, logvar


# =========================================================================
# definition of latent class
# 
# DIMENSION NOTE:
# [input]   : dimension of input is ((N, dim_latent, 1, 1), (N, dim_latent, 1, 1))
# [output]  : dimension of output is (N, dim_latent, 1, 1)
# =========================================================================
class MyLatent(nn.Module):
    def __init__(self, dim_latent: int=100):
        super(MyLatent, self).__init__()
        self.dim_latent = dim_latent


    def forward(self, mu, logvar):
        sigma   = torch.exp(0.5 * logvar)
        epsilon = self.sample_latent(sigma.size(0))
        z       = mu + sigma * epsilon
        return z 


    def sample_latent(self, num_latent: int=10):
        z = torch.randn(num_latent, self.dim_latent, 1, 1)
        return z


# =========================================================================
# definition of Decoder
# 
# DIMENSION NOTE:
# [input]   : dimension of input is (N, dim_latent, 1, 1)
# [output]  : dimension of output is (N, dim_channel, dim_height, dim_width)
# =========================================================================
class MyDecoder(nn.Module):
    def __init__(self, dim_latent: int=100, dim_channel: int=1, dim_height: int=32, dim_width: int=32):
        super(MyDecoder, self).__init__()
        self.dim_latent     = dim_latent  
        self.dim_channel    = dim_channel
        self.dim_height     = dim_height
        self.dim_width      = dim_width

        conv_out_h = 4
        conv_out_w = 4  
        self.conv_out_h = conv_out_h
        self.conv_out_w = conv_out_w

        self.fc = nn.Linear(dim_latent,512 * 4 * 4)

        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, dim_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.conv_out_h, self.conv_out_w)
        
        x = self.pipe(x)

        return x
    
