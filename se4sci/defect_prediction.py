"""
A Defect Prediction model for file level metrics
"""
import os
import re
import sys
import pandas as pd
from pdb import set_trace
from prettytable import PrettyTable

from pathlib import Path
root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from metrics.abcd import ABCD
from data.data_handler import DataHandler
from prediction.model import PredictionModel


if __name__ == "__main__":
    dh = DataHandler()
    mdl = PredictionModel()
    data = dh.get_data()
    
    "Create a Table than can pretty printed"
    results = PrettyTable()
    results.field_names = ["Train", "Test ", "   Pd", "   Pf", "   F1"]
    
    "Align Data"
    results.align["Train"] = "l"
    results.align["Test "] = "l"
    results.align["   Pd"] = "r"
    results.align["   Pf"] = "r"
    results.align["   F1"] = "r"
    
    for proj, dataset in data.items():
        dataset_keys = sorted(dataset.keys())
        for trn, tst in zip(dataset_keys[:-1], dataset_keys[1:]):
            train = dataset[trn]
            test = dataset[tst]
            try:
                actual, predicted = mdl.predict_defects(
                    train, test, oversample=True)
            except ValueError:  
                "Data issue, if all class values are the same."
                continue
            pd, pf = ABCD.get_pd_pf(actual, predicted)
            _, __, f = ABCD.get_f_score(actual, predicted)
            trn_name = re.sub("file_metrics_|.csv", "", trn)
            tst_name = re.sub("file_metrics_|.csv", "", tst)
            results.add_row([trn_name, tst_name, pd, pf, f])

        print(results)
