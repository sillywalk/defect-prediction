import os
import sys
import numpy
import unittest
from pathlib import Path
from pdb import set_trace

root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from metrics.abcd import ABCD


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)

    def test_pd_pf(self):
        actual = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
        predicted = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        pd, pf = ABCD.get_pd_pf(actual, predicted)
        self.assertAlmostEqual(pd, 0.71, places=2)
        self.assertAlmostEqual(pf, 0.43, places=2)

    def test_f_score(self):
        actual = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
        predicted = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        _, __, f1 = ABCD.get_f_score(actual, predicted, beta=1)
        _, __, f2 = ABCD.get_f_score(actual, predicted, beta=2)
        self.assertAlmostEqual(f1, 0.66, places=2)
        self.assertAlmostEqual(f2, 0.69, places=2)
