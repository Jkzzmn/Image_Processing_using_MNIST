import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from MyModel import MyModel
from torchmetrics.image import PeakSignalNoiseRatio

class MyEval():
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio(data_range=8.0)    # data range = [-1.0, 1.0], noise range = [-3.0, 3.0] (std = 0.5), so the data range = 8.0

    # value range of prediction: 0 ~ 1
    # value range of label: 0 or 1
    # accuracy range: 0 ~ 1 
    def compute_accuracy(self, prediction, label):
        pred_label = torch.argmax(prediction, dim=1)
        pred_label = pred_label.to(torch.uint8)
        label      = label.to(torch.uint8)
        bCorrect    = (pred_label == label)
        accuracy    = bCorrect.sum() / len(label)
        # print(f'prediction: {pred_label}')
        # print(f'label: {label}')
        # print(f'correct: {bCorrect}, accuracy: {accuracy}')
        return accuracy

    def psnr_reset(self):
        self.psnr.reset()
    
    def psnr_update(self, prediction, target):
        self.psnr.update(prediction, target)
    
    def psnr_compute(self):
        psnr_value = self.psnr.compute()
        return psnr_value.item()
