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
        if split.lower() == 'train':
            self.dir_data   = os.path.join(path, split)
            self.files      = glob.glob(self.dir_data + '/*.png')
            if transform is None:
                self.transform  = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            else:
                self.transform = transform
        elif split.lower() == 'real':
            self.dir_data   = os.path.join(path, 'train')
            self.files      = glob.glob(self.dir_data + '/*.png')
            self.transform  = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(299, antialias=True), transforms.ToTensor()])
        elif split.lower() == 'fake':
            self.dir_data   = os.path.join(path, 'sample')
            self.files      = glob.glob(self.dir_data + '/*.png')
            self.transform  = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(299, antialias=True), transforms.ToTensor()])
        else:
            raise NotImplementedError('Only "train", "real", and "fake" splits are implemented.')


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_data = self.files[idx]
        data = Image.open(file_data)

        if self.transform is not None:
            data = self.transform(data)

        return data

