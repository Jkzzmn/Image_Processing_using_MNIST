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
        dir_dataset     = os.path.join(path, split)
        self.transform  = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]) if transform is None else transform
        self.dataset    = datasets.ImageFolder(root=dir_dataset, transform=self.transform)

    def __new__(cls, path: str='data', split: str='train', transform=None):
        # if split not in ['train', 'val', 'test']:
        #     raise ValueError("split must be 'train', 'val', or 'test'")
        instance = super(MyDataset, cls).__new__(cls)
        instance.__init__(path, split, transform)
        return instance.dataset

# =========================================================================
if __name__ == '__main__':
    dataset     = MyDataset(path='data', split='train')
    dataloader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    iter_data   = iter(dataloader)
    data, label = next(iter_data)
    vec         = nn.Flatten()(data)
    print(f'dataset: {len(dataset)}')
    print(f'data: {data.shape}')
    print(f'label: {label.shape}')
    print(f'vec: {vec.shape}')

    plt.imshow(data[0].squeeze(), cmap='gray')
    plt.title(f'label: {label.item()}')
    plt.show()