import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestExample(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1,1)

