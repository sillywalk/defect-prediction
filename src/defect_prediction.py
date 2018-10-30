"""
A Defect Prediction model for file level metrics
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from pdb import set_trace
from prettytable import PrettyTable


from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))

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
    results.field_names = [
        "Train", " Test", " Prec", "   Pd", "   Pf", "   F1", "  IFA", "PCI20"]

    "Align Data"
    results.align[""] = "l"
    results.align["   Pd"] = "r"
    results.align["   Pd"] = "r"
    results.align["   Pf"] = "r"
    results.align["   Pf"] = "r"
    results.align["   F1"] = "r"
    results.align["   F1"] = "r"
    for proj, dataset in data.items():
        print(proj.upper())
        i = 0
        print("Train", "Test", "Prec", "Pd", "Pf",
            "F1", "IFA", "PCI20", sep="\t")
        for train_dataframe, test_dataframe in zip(dataset[:-1], dataset[1:]):
            p_d = []
            p_f = []
            f_1 = []
            ifa = [] 
            prec = []
            pci20 = []
            pred_mod = PredictionModel(classifier='SVC')
            for _ in range(10):
                # Make predictions
                actual, predicted = pred_mod.predict_defects(
                    train_dataframe, test_dataframe)
                
                # Get lines of code of the test instances
                loc = test_dataframe["CountLine"]
                abcd = ABCD(actual, predicted, loc)
        
                # Get performance metrics
                ifa_obtained = abcd.get_ifa()
                pci20_obtained = abcd.get_pci_20()
                pd_obtained, pf_obtained = abcd.get_pd_pf()
                prec_obtained, f1_obtained = abcd.get_f_score()

                # Gather the obtained performance metrics
                p_d.append(pd_obtained)
                p_f.append(pf_obtained)
                f_1.append(f1_obtained)
                ifa.append(ifa_obtained)
                prec.append(prec_obtained)
                pci20.append(pci20_obtained)
                
            i += 1
            print(i, i+1, 
                  np.median(prec).astype("int"),
                  np.median(p_d).astype("int"),
                  np.median(p_f).astype("int"),
                  np.median(f_1).astype("int"),
                  np.median(ifa).astype("int"),
                  np.median(pci20).astype("int"),
                  sep="\t")
        
        print("\n"+50*"-"+"\n")
