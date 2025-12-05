import unittest
import os
import numpy as np
import csv
import sys

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import statistics

from MyDataset import MyDataset
from MyEval import MyEval
from MyModel import MyModel
from MyResult import MyResult
from MyTrain import MyTrain


class TestResult(unittest.TestCase):
    model       = MyModel()
    eval        = MyEval(metric='iou')
    result      = MyResult()
    trainer     = MyTrain(model=model)

    batch_size  = 10
    file_result = 'result_autograding.csv'
    path_data   = 'data'

    # setUpClass is called once 
    @classmethod
    def setUpClass(cls):
        pass

    # setUp is called before each test method
    def setUp(self):
        pass 

    def test_1_model_train(self):
        dataset = MyDataset(path=self.path_data, split='train')
        print(f'training on the dataset (train): {len(dataset)}')
        self.trainer.train(dataset=dataset)
        self.trainer.get_model().save()
        
    def test_2_model_test(self):
        dataset      = MyDataset(path=self.path_data, split='test')
        dataloader   = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        model        = self.trainer.get_model()
        model.eval()

        print(f'testing on the dataset (test): {len(dataset)}')
        self.eval.metric_reset()
        for step, (image, mask) in enumerate(dataloader):
            print(f'Testing step {step+1}/{len(dataloader)}...')
            pred = model(image)
            pred = (pred > 0.5).float()  # Binarize the prediction
       
            sum1 = (pred * mask).sum(dim=(1,2,3))       # size: (batch_size,)
            sum2 = ((1 - pred) * mask).sum(dim=(1,2,3)) # size: (batch_size,)
            idx2 = (sum1 < sum2)                       # size: (batch_size,)
            pred[idx2] = 1 - pred[idx2]
                   
            self.eval.metric_update(pred, mask)
       
        metric_value = self.eval.metric_compute()
        self.result.add_result('test', metric_value)
        print(f'Metric (test): {metric_value}')
    # tearDown is called after each test method 
    def tearDown(self):
        pass

    # tearDownClass is called once 
    @classmethod
    def tearDownClass(cls):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('result on the testing data: ', cls.result.get_result('test'))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        cls.result.save(cls.file_result)


if __name__ == '__main__':
    unittest.main()
