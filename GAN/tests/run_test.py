import unittest
import csv
import sys

import os
from pyexpat import model
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import statistics
import numpy as np
from torchvision.utils import make_grid 
from torchvision.utils import save_image

from torchvision import datasets
from torch.utils.data import Dataset

from MyDataset import MyDataset
from MyEval import MyEval
from MyModel import MyDiscriminator, MyGenerator
from MyResult import MyResult
from MyTrain import MyTrain
from MySample import MySample


class TestResult(unittest.TestCase):

    # setUpClass is called once 
    @classmethod
    def setUpClass(cls):
        cls.generator   = MyGenerator(1, 32, 32)
        state_dict      = torch.load('generator.pth', map_location=torch.device('cpu'))
        cls.generator.load_state_dict(state_dict)
        cls.sampler     = MySample(cls.generator)
        cls.eval        = MyEval(metric='fid')
        cls.result      = MyResult()
        cls.file_result = 'result_autograding.csv'
        cls.path_sample = 'data/sample'

    # setUp is called before each test method
    def setUp(self):
        pass 

    def test_1_model_train(self):
        # print('training the model...')
        pass
        
    def test_2_model_test(self):
        num_sample  = 100
        batch_size  = 10
        num_batch   = num_sample // batch_size
        list_sample = []
        
        print(f'===============================================')
        print(f'generate samples...') 
        print(f'===============================================')
        for i in range(num_batch):
            print(f'generating batch {i+1}/{num_batch}...')
            sample = self.sampler.sample(batch_size)
            list_sample.append(sample)
       
        samples = torch.cat(list_sample, dim=0)
        
        print(f'===============================================')
        print(f'save samples...') 
        print(f'===============================================')
        for j in range(samples.size(0)):
            print(f'saving sample {j+1}/{samples.size(0)}...')
            file_save = os.path.join(self.path_sample, f'{j:03d}.png')
            save_image(samples[j], file_save)
            
        print(f'===============================================')
        print(f'computing FID...') 
        print(f'===============================================')
        fid = self.eval.compute()
        self.result.add_result('fid', fid)

    # tearDown is called after each test method 
    def tearDown(self):
        pass

    # tearDownClass is called once 
    @classmethod
    def tearDownClass(cls):
        print(f'===============================================')
        print(f'FID: {cls.result.get_result("fid")}') 
        print(f'===============================================')
        cls.result.save(cls.file_result)

if __name__ == '__main__':
    unittest.main()
