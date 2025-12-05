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
    model   = MyModel()
    eval    = MyEval()
    result  = MyResult()

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
        dataloader  = DataLoader(dataset, batch_size=10, drop_last=False, shuffle=False)
        print(f'dataset (train): {len(dataset)}')
       
        result_batch = []

        for step, (data, label) in enumerate(dataloader):
            pred    = self.model(data)
            acc     = self.eval.compute_accuracy(pred, label)
            result_batch.append(acc.item())

        result_mean = statistics.mean(result_batch)
        print(f'result (train): {result_mean}')
        self.result.add_result('train', result_mean)

    def test_3_eval_test(self):
        dataset     = MyDataset(path='data', split='test')
        dataloader  = DataLoader(dataset, batch_size=10, drop_last=False, shuffle=False)
        print(f'dataset (test): {len(dataset)}')
       
        result_batch = []
        
        for step, (data, label) in enumerate(dataloader):
            pred    = self.model(data)
            acc     = self.eval.compute_accuracy(pred, label)
            result_batch.append(acc.item())

        result_mean = statistics.mean(result_batch)
        print(f'result (test): {result_mean}')
        self.result.add_result('test', result_mean)

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
        cls.result.save('result_autograding.csv')


if __name__ == '__main__':
    unittest.main()