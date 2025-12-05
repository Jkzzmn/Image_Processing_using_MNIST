import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm 

class MyTrain():
    def __init__(self,model,data_loader):
        self.model = model
        self.data_loader = data_loader
        pass



