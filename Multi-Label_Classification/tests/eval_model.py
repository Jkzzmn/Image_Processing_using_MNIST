import unittest
import os
import csv
import sys
import argparse
from MyResult import MyResult

class TestResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        result = MyResult()
        result.load('result_autograding.csv')
        cls.key     = 'load'
        cls.value   = 1.0 
        cls.result  = result.get_result(cls.key)

    def setUp(self):
        pass 

    def test_result(self):
        self.assertEqual(self.result, self.value, f"{self.result} should be equal to {self.value}")

    def tearDown(self):
        pass

    # tearDownClass is called once 
    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()