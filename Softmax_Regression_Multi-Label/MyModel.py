import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        in_channels     = 1
        out_channels    = 16
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.bn   = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten    = nn.Flatten()
        self.linear     = nn.Linear(16*8*8, 10)

    def forward(self, x):
        x = self.conv(x)

        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.linear(x)
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


if __name__ == '__main__':
    dataset     = MyDataset(path='data', split='train')
    dataloader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    iter_data   = iter(dataloader)
    data, label = next(iter_data)
    vec         = nn.Flatten()(data)

    model1  = MyModel()
    model1.eval()
    model1.save()
    
    model2  = MyModel()
    model2.load()
    model2.eval()
    
    y1 = model1(vec)
    y2 = model2(vec)
    
    print(f'dataset: {len(dataset)}')
    print(f'input: {vec.shape}, y1: {y1.shape}, y1: {y1}')
    print(f'input: {vec.shape}, y2: {y2.shape}, y2: {y2}')
    print(f'y1 == y2: {torch.equal(y1, y2)}')
    print(f'model1.size(): {model1.size()}')
    print(f'model2.size(): {model2.size()}')