import os
import sys
import numpy
import unittest
from pathlib import Path
from pdb import set_trace
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from data.data_handler import DataHandler
from prediction.model import PredictionModel

class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.dh = DataHandler()
        self.mdl = PredictionModel()

    def test_prediction_model(self):
        data = self.dh.get_data()
        for proj, dataset in data.items():
            dataset_keys = sorted(dataset.keys())
            for trn, tst in zip(dataset_keys[:-1], dataset_keys[1:]):
                train = dataset[trn]
                test = dataset[tst]
                try:
                    actual, predicted = self.mdl.predict_defects(
                        train, test, oversample=True)
                except ValueError:
                    "Data issue, if all class values are the same."
                    continue

                self.assertIsInstance(actual, numpy.ndarray)
                self.assertIsInstance(predicted, numpy.ndarray)
                self.assertEqual(len(actual), len(predicted))
