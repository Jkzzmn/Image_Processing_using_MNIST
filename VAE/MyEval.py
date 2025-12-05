import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from MyDataset import MyDataset
from torch.utils.data import Dataset, DataLoader

class MyEval():
    def __init__(self, metric: str='fid'):
        if metric.lower() == 'fid':
            self.metric = FrechetInceptionDistance(normalize=True)
        else:
            raise NotImplementedError('Only "fid" metric is implemented.')
        self.batch_size = 10


    def compute(self):
        self.metric.reset()
        self.update_real()
        self.update_fake()
        metric_value = self.metric.compute()
        return metric_value.item()


    def update_real(self, path: str='data', split: str='real'):
        print(f'===============================================')
        print(f'update real dataset...') 
        print(f'===============================================')
        dataset_real    = MyDataset(path=path, split=split)
        dataloader_real = DataLoader(dataset_real, batch_size=self.batch_size, shuffle=False)
        for image_real in dataloader_real:
            self.metric.update(image_real, real=True)


    def update_fake(self, path: str='data', split: str='fake'):
        print(f'===============================================')
        print(f'update fake dataset...') 
        print(f'===============================================')
        dataset_fake    = MyDataset(path=path, split=split)
        dataloader_fake = DataLoader(dataset_fake, batch_size=self.batch_size, shuffle=False)
        for image_fake in dataloader_fake:  
            self.metric.update(image_fake, real=False)
