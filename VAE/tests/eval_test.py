import unittest
import os
import csv
import sys
import argparse
from MyResult import MyResult

class TestResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result          = MyResult()
        result.load('result_autograding.csv')
        key             = 'fid'
        cls.result      = result.get_result(key)
        cls.threshold   = 50 
        
    def setUp(self):
        pass 

    def test_result(self):
        self.assertLess(self.result, self.threshold, f"{self.result} should be less than {self.threshold}")

    def tearDown(self):
        pass

    # tearDownClass is called once 
    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()