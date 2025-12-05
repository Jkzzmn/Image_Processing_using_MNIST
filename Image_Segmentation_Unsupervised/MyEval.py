import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from MyModel import MyModel
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.classification import JaccardIndex, MulticlassAccuracy # IoU is alias for JaccardIndex

class MyEval():
    def __init__(self, metric: str='psnr'):
        if metric.lower() == 'psnr':
            self.metric = PeakSignalNoiseRatio(data_range=8.0)    # data range = [-1.0, 1.0], noise range = [-3.0, 3.0] (std = 0.5), so the data range = 8.0
        elif metric.lower() == 'iou':
            self.metric = JaccardIndex(task="binary", num_classes=2)
        elif metric.lower() == 'accuracy':
            self.metric = MulticlassAccuracy(task="multiclass", num_classes=5)
        else:
            raise ValueError(f'Unsupported metric: {metric}')
        
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

    def metric_reset(self):
        self.metric.reset()
    
    def metric_update(self, prediction, target):
        self.metric.update(prediction, target)
    
    def metric_compute(self):
        metric_value = self.metric.compute()
        return metric_value.item()
    