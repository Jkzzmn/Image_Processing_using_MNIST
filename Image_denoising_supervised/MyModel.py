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
            nn.Conv2d(in_channels,32,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,padding=1,stride=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(32,out_channels,kernel_size=3,padding=1,stride=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
