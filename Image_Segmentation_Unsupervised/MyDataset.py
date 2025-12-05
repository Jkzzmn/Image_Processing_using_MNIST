import os
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import GaussianNoise
import glob


class MyDataset(Dataset):
    def __init__(self, path: str='data', split: str='train'):
        self.dir_data   = os.path.join(path, split)
        # =========================================================================
        # define your own transforms for data augmentation
        # =========================================================================
        if split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),                            # data range = [0.0, 1.0]
            ])
            self.file_image = glob.glob(self.dir_data + '/image/*.png')
            self.file_mask  = glob.glob(self.dir_data + '/image/*.png')
        # =========================================================================
        # do not modify the following codes
        # =========================================================================
        elif split.lower() == 'test':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),              # data range = [0.0, 1.0]
            ])
            self.file_image = glob.glob(self.dir_data + '/image/*.png')
            self.file_mask  = glob.glob(self.dir_data + '/mask/*.png')


    def __len__(self):
        return len(self.file_image)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_image      = self.file_image[idx]
        file_mask       = self.file_mask[idx]
        image           = Image.open(file_image)
        mask            = Image.open(file_mask)

        if self.transform is not None:
            image = self.transform(image)
            mask  = self.transform(mask)

        return (image, mask)
    
    '''
    def __new__(cls, path: str='data', split: str='train', transform=None):
        instance = super(MyDataset, cls).__new__(cls)
        instance.__init__(path, split, transform)
        return instance
    '''