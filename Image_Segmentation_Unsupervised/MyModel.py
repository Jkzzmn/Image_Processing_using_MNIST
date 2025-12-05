import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset

class MyModel(nn.Module):
    # =========================================================================
    # define your own network architecture
    # =========================================================================
    def __init__(self):
        super(MyModel, self).__init__()
        in_channels     = 1
        out_channels    = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        #x = self.conv(x)
        #x = self.activation(x)
        #x = self.conv_trans(x)
        #x = self.activation(x)
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x

    # =========================================================================
    # do not modify the following codes
    # =========================================================================
    def save(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        self.load_state_dict(torch.load(path))

    def size(self):
        size = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return size
    
    def print(self):
        print(self.state_dict())
    # =========================================================================
