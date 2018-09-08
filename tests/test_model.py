import os
import sys
import unittest
from pathlib import Path
from pdb import set_trace

root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from data.data_handler import DataHandler
from defect_predict.model import PredictionModel

class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.dh = DataHandler()
        self.mdl = PredictionModel()

    def test_prediction_model(self):
        data = self.dh.get_data()
        for proj, datasets in data.items():
            set_trace()