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


class TestResult(unittest.TestCase):
    model       = MyModel()
    eval        = MyEval(metric='iou')
    result      = MyResult()
    batch_size  = 10
    file_result = 'result_autograding.csv'
    # setUpClass is called once 
    @classmethod
    def setUpClass(cls):
        pass

    # setUp is called before each test method
    def setUp(self):
        pass 

    def test_1_load_model(self):
        try:
            self.model.load()
            self.result.add_result('load', 1.0)
            print(f'model is loaded.')
        except FileNotFoundError:
            self.result.add_result('load', 0.0)
            print("Error: Checkpoint file not found.")

        self.model.eval()
        
    def test_2_eval_train(self):
        dataset     = MyDataset(path='data', split='train')
        dataloader  = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        print(f'dataset (train): {len(dataset)}')
       
        self.eval.metric_reset()
        for step, (image, mask) in enumerate(dataloader):
            pred = self.model(image)
            pred = (pred > 0.5).float()  # Binarize the prediction
            self.eval.metric_update(pred, mask)
        metric_value = self.eval.metric_compute()
        self.result.add_result('train', metric_value)
        print(f'Metric (train): {metric_value}')

    def test_3_eval_test(self):
        dataset     = MyDataset(path='data', split='test')
        dataloader  = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        print(f'dataset (test): {len(dataset)}')
       
        self.eval.metric_reset()
        for step, (image, mask) in enumerate(dataloader):
            pred = self.model(image)
            pred = (pred > 0.5).float()  # Binarize the prediction
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
        print('model is loaded: ', cls.result.get_result('load'))
        print('result on the training data: ', cls.result.get_result('train'))
        print('result on the testing data: ', cls.result.get_result('test'))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        cls.result.save(cls.file_result)


if __name__ == '__main__':
    unittest.main()