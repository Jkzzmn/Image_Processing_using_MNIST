import unittest
import os
import csv
import sys
import argparse
from MyResult import MyResult

class TestResult(unittest.TestCase):
    key        = 'test'
    threshold  = 10.0

    @classmethod
    def setUpClass(cls):
        result = MyResult()
        result.load('result_autograding.csv')
        cls.result = result.get_result(cls.key)

    def setUp(self):
        pass 

    def test_result(self):
        self.assertGreater(self.result, self.threshold, f"{self.result} is less than or equal to {self.threshold}")

    def tearDown(self):
        pass

    # tearDownClass is called once 
    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()