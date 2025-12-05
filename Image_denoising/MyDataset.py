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

class MyDataset(Dataset):
    def __init__(self, path: str='data', split: str='train', transform=None):
        # =========================================================================
        # define your own transforms for data augmentation
        # =========================================================================
        if split.lower() == 'mytrain':
            split = 'train'
            self.transform  = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),              # data range = [0.0, 1.0]
                transforms.Normalize([0.5], [0.5]), # data range = [-1.0, 1.0]
                transforms.RandomHorizontalFlip(),  
                transforms.RandomVerticalFlip(),   
                transforms.RandomRotation(15),     
            ]) if transform is None else transform
        # =========================================================================
        # do not modify the following codes
        # =========================================================================
        else:
            self.transform  = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),              # data range = [0.0, 1.0]
                transforms.Normalize([0.5], [0.5]), # data range = [-1.0, 1.0]
            ]) if transform is None else transform

        dir_dataset     = os.path.join(path, split)
        self.dataset    = datasets.ImageFolder(root=dir_dataset, transform=self.transform)

    def __new__(cls, path: str='data', split: str='train', transform=None):
        instance = super(MyDataset, cls).__new__(cls)
        instance.__init__(path, split, transform)
        return instance.dataset
