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
import warnings
warnings.filterwarnings("ignore")
import pickle
import time



from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
print(root)
if root not in sys.path:
    sys.path.append(str(root))

from metrics.abcd import ABCD
from data.data_handler import DataHandler
from prediction.model import PredictionModel


projects =['libmesh', 'mdanalysis', 'lammps', 'abinit']
data_collections = ['fastread_jit_file', 'release_level', 'keyword_jit_file', 'human_jit_file']

def release_base_running():
    return None


def execute(filename, file_type):
    dh = DataHandler()
    mdl = PredictionModel()
    data = dh.get_data()
    set_trace()
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
    records = {}
    for proj in projects:
        print(proj.upper())

        for type_p in data_collections:
            if type_p not in records.keys():
                records[type_p] = {}
            records[type_p][proj] = {"Prec": [], "Pd": [], "Pf": [],
                "F1":[], "IFA":[], "PCI20":[],
                "Prec_all":[], "F1_all":[], "Time": []}
            dataset = data[type_p][proj]
        #for proj, dataset in projects.items():
            print(type_p.upper(), str(len(dataset)) + " releases, ", end= "")
            no_datasets = len(dataset)
            #print(#"Train", "Test",
            #      "Prec", "Pd", "Pf",
            #    "F1", "IFA", "PCI20",
            #    "Prec_all", "F1_all",
            #    sep=",\t")
            for i in range(no_datasets-1):
                print(" -> ", i, end="")
                #train_dataframe = dataset[i]
                if i == 0:
                    train_dataframe = dataset[i]
                else:
                    train_dataframe = pd.concat(dataset[:i+1])
                test_dataframe = data['human_jit_file'][proj][i + 1]
                if type_p == "release_level":
                    train_dataframe = train_dataframe.drop(['fastread_bugs', 'human_bugs'], axis=1)
                    #print(len(train_dataframe.columns.tolist()), len(test_dataframe.columns.tolist()))

                #set_trace()
                p_d, p_f = [], []
                f_1, prec = [], []
                ifa , pci20 = [], []
                total_prec, total_f1 = [], []
                running_times = []
                pred_mod = PredictionModel(classifier='SVC')
                for _ in range(10):
                    # Make predictions
                    start_time = time.time()
                    actual, predicted = pred_mod.predict_defects(
                        train_dataframe, test_dataframe)
                    delta_time = time.time() - start_time

                    # Get lines of code of the test instances
                    loc = test_dataframe["CountLine"]
                    abcd = ABCD(actual, predicted, loc)

                    # Get performance metrics
                    ifa_obtained = abcd.get_ifa()
                    pci20_obtained = abcd.get_pci_20()
                    pd_obtained, pf_obtained = abcd.get_pd_pf()
                    prec_obtained, f1_obtained = abcd.get_f_score()
                    total_prec_val, total_f1_val = abcd.get_score_total()


                    # Gather the obtained performance metrics
                    p_d.append(pd_obtained)
                    p_f.append(pf_obtained)
                    f_1.append(f1_obtained)
                    ifa.append(ifa_obtained)
                    prec.append(prec_obtained)
                    pci20.append(pci20_obtained)
                    total_prec.append(total_prec_val)
                    total_f1.append(total_f1_val)
                    running_times.append(delta_time)

                i += 1
                little_res = [np.median(prec).astype("int"),
                              np.median(p_d).astype("int"),
                              np.median(p_f).astype("int"),
                              np.median(f_1).astype("int"),
                              np.median(ifa).astype("int"),
                              np.median(pci20).astype("int"),
                              np.median(total_prec).astype("int"),
                              np.median(total_f1).astype("int"),
                              np.median(running_times)]
                records[type_p][proj]['Prec'].append(little_res[0])
                records[type_p][proj]['Pd'].append(little_res[1])
                records[type_p][proj]['Pf'].append(little_res[2])
                records[type_p][proj]['F1'].append(little_res[3])
                records[type_p][proj]['IFA'].append(little_res[4])
                records[type_p][proj]['PCI20'].append(little_res[5])
                records[type_p][proj]['Prec_all'].append(little_res[6])
                records[type_p][proj]['F1_all'].append(little_res[7])
                records[type_p][proj]['Time'].append(little_res[8])



                #print(#i, i+1,
                #      little_res,
                #      sep=",\t")
            print()
            set_trace()
        #filehandler = open(filename, "wb")
        #pickle.dump(records, filehandler)
        #filehandler.close()
        for type_p in data_collections:
            dataset = data[type_p][proj]
            no_datasets = len(dataset)
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf",
                "F1", "IFA", "PCI20",
                "Prec_all", "F1_all", "Time",
                sep=",\t")
            if type_p != "release_level":
                for i in range(no_datasets - 1):
                    little_c = [records[file_type][proj]['Prec'][i] - records[type_p][proj]['Prec'][i],
                                records[file_type][proj]['Pd'][i] - records[type_p][proj]['Pd'][i],
                                records[file_type][proj]['Pf'][i] - records[type_p][proj]['Pf'][i],
                                records[file_type][proj]['F1'][i] - records[type_p][proj]['F1'][i],
                                records[file_type][proj]['IFA'][i] - records[type_p][proj]['IFA'][i],
                                records[file_type][proj]['PCI20'][i] - records[type_p][proj]['PCI20'][i],
                                records[file_type][proj]['Prec_all'][i] - records[type_p][proj]['Prec_all'][i],
                                records[file_type][proj]['F1_all'][i] - records[type_p][proj]['F1_all'][i],
                                records[type_p][proj]['Time'][i]]
                    for j, m in zip([0, 1, 2, 3, 6, 7], ['Prec', 'Pd', 'Pf', 'F1', 'Prec_all', 'F1_all']):
                        if records[type_p][proj][m][i] != 0:
                            little_c[j] = (float(little_c[j]) * 100) / (records[type_p][proj][m][i])
                        else:
                            little_c[j] = little_c[j]
                    print(  # i, i+1,
                        int(little_c[0]), int(little_c[1]), int(little_c[2]), int(little_c[3]),
                        little_c[4], little_c[5], int(little_c[6]), int(little_c[7]), round(little_c[8], 3),
                    sep = ",\t")
            else:
                for i in range(no_datasets - 1):
                    print(records[type_p][proj]['Prec'][i],
                          records[type_p][proj]['Pd'][i], records[type_p][proj]['Pf'][i],
                          records[type_p][proj]['F1'][i], records[type_p][proj]['IFA'][i],
                          records[type_p][proj]['PCI20'][i], records[type_p][proj]['Prec_all'][i],
                          records[type_p][proj]['F1_all'][i], round(records[type_p][proj]['Time'][i],3),
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


if __name__ == "__main__":
    file_name = sys.argv[1]
    file_type = sys.argv[2]
    print(file_name, file_type)
    execute(file_name, file_type)
