"""
A Defect Prediction model for file level metrics
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
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


projects =['mdanalysis', 'libmesh', 'lammps', 'abinit']
data_collections = ['fastread_jit_file', 'release_level', 'keyword_jit_file', 'human_jit_file']

def release_base_running():
    return None


def execute(filename, file_type, level, learner, sampling, all):
    dh = DataHandler(level)
    data = dh.get_data()
    #set_trace()
    records = {}

    if os.path.isfile(filename):
        filehandler = open(filename, "rb")
        try:
            results_records_file = pickle.load(filehandler)
        except EOFError:
            results_records_file = []
        filehandler.close()


    for proj in projects:
        print(proj.upper())
        for type_p in data.keys():
            if type_p not in records.keys():
                records[type_p] = {}
            records[type_p][proj] = {"Prec": [], "Pd": [], "Pf": [], "F1":[],  "G1": [],
                                     "AUC_Prec": [], "AUC_Pd": [], "AUC_Pf": [], "AUC_F1": [], "AUC_G1": [],
                                     "IFA":[], "PCI20":[],
                "Prec_all":[], "F1_all":[], "Time": []}
            dataset = data[type_p][proj]

            print(type_p.upper(), str(len(dataset)) + " releases, ", end= "")
            no_datasets = len(dataset)
            for i in range(no_datasets-1):
                print(" -> ", i, end="")
                if all == 1:
                    #print("all")
                    if i == 0:
                        train_dataframe = dataset[i]
                    else:
                        train_dataframe = pd.concat(dataset[:i+1])
                else:
                    #print("incremental")
                    train_dataframe = dataset[i]
                if level == "file":
                    test_dataframe = data['human_jit_file'][proj][i + 1]
                    # Get lines of code of the test instances
                    loc = test_dataframe["CountLine"]
                    if type_p == "release_level":
                        train_dataframe = train_dataframe.drop(['fastread_bugs', 'human_bugs'], axis=1)
                else:
                    test_dataframe = data['human'][proj][i + 1]
                    loc = test_dataframe["lt"]*test_dataframe["nf"]

                #set_trace()
                p_d, p_f = [], []
                f_1, prec = [], []
                ifa , pci20 = [], []
                g_1 = []
                auc_pd, auc_pf, auc_prec, auc_f1, auc_g1 = [], [], [], [], []
                total_prec, total_f1 = [], []
                running_times = []
                pred_mod = PredictionModel(classifier=learner)
                for _ in range(10):
                    # Make predictions
                    start_time = time.time()
                    actual, predicted = pred_mod.predict_defects(
                        train_dataframe, test_dataframe, samplingtechnique=sampling)
                    delta_time = time.time() - start_time
                    abcd = ABCD(actual, predicted, loc)

                    # Get performance metrics
                    ifa_obtained = abcd.get_ifa()
                    pci20_obtained = abcd.get_pci_20()
                    pd_obtained, pf_obtained = abcd.get_pd_pf()
                    prec_obtained, f1_obtained = abcd.get_f_score()
                    g1_obtained = abcd.get_G()
                    auc_pd_val, auc_pf_val, auc_prec_val, auc_f1_val, auc_g1_val = abcd._set_auc(3)
                    total_prec_val, total_f1_val = abcd.get_score_total()


                    # Gather the obtained performance metrics
                    p_d.append(pd_obtained)
                    p_f.append(pf_obtained)
                    f_1.append(f1_obtained)
                    g_1.append(g1_obtained)
                    auc_pd.append(auc_pd_val)
                    auc_pf.append(auc_pf_val)
                    auc_prec.append(auc_prec_val)
                    auc_f1.append(auc_f1_val)
                    auc_g1.append(auc_g1_val)
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
                              np.median(g_1).astype("int"),
                              np.median(auc_prec).astype("int"),
                              np.median(auc_pd).astype("int"),
                              np.median(auc_pf).astype("int"),
                              np.median(auc_f1).astype("int"),
                              np.median(auc_g1).astype("int"),
                              np.median(ifa).astype("int"),
                              np.median(pci20).astype("int"),
                              np.median(total_prec).astype("int"),
                              np.median(total_f1).astype("int")]
                              #np.median(running_times)]
                records[type_p][proj]['Prec'].append(little_res[0])
                records[type_p][proj]['Pd'].append(little_res[1])
                records[type_p][proj]['Pf'].append(little_res[2])
                records[type_p][proj]['F1'].append(little_res[3])
                records[type_p][proj]['G1'].append(little_res[4])
                records[type_p][proj]['AUC_Prec'].append(little_res[5])
                records[type_p][proj]['AUC_Pd'].append(little_res[6])
                records[type_p][proj]['AUC_Pf'].append(little_res[7])
                records[type_p][proj]['AUC_F1'].append(little_res[8])
                records[type_p][proj]['AUC_G1'].append(little_res[9])
                records[type_p][proj]['IFA'].append(little_res[10])
                records[type_p][proj]['PCI20'].append(little_res[11])
                records[type_p][proj]['Prec_all'].append(little_res[12])
                records[type_p][proj]['F1_all'].append(little_res[13])
                #records[type_p][proj]['Time'].append(little_res[9])



                #print(#i, i+1,
                #      little_res,
                #      sep=",\t")
            print()
            #set_trace()
        print("save + ", filename)
        filehandler = open(filename, "wb")
        pickle.dump(records, filehandler)
        for type_p in data.keys():
            dataset = data[type_p][proj]
            no_datasets = len(dataset)
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                "AUC_Pr", "AUC_Pd", "AUC_Pf", "AUC_F1", "AUC_G1",
                "IFA", "PCI20", "Pr_all", "F1_all",
                #"Time",
                sep=",\t")
            if type_p != file_type:
                for i in range(no_datasets - 1):
                    little_c = [records[file_type][proj]['Prec'][i] - records[type_p][proj]['Prec'][i],
                                records[file_type][proj]['Pd'][i] - records[type_p][proj]['Pd'][i],
                                records[file_type][proj]['Pf'][i] - records[type_p][proj]['Pf'][i],
                                records[file_type][proj]['F1'][i] - records[type_p][proj]['F1'][i],
                                records[file_type][proj]['G1'][i] - records[type_p][proj]['G1'][i],
                                records[file_type][proj]['AUC_Prec'][i] - records[type_p][proj]['AUC_Prec'][i],
                                records[file_type][proj]['AUC_Pd'][i] - records[type_p][proj]['AUC_Pd'][i],
                                records[file_type][proj]['AUC_Pf'][i] - records[type_p][proj]['AUC_Pf'][i],
                                records[file_type][proj]['AUC_F1'][i] - records[type_p][proj]['AUC_F1'][i],
                                records[file_type][proj]['AUC_G1'][i] - records[type_p][proj]['AUC_G1'][i],
                                records[file_type][proj]['IFA'][i] - records[type_p][proj]['IFA'][i],
                                records[file_type][proj]['PCI20'][i] - records[type_p][proj]['PCI20'][i],
                                records[file_type][proj]['Prec_all'][i] - records[type_p][proj]['Prec_all'][i],
                                records[file_type][proj]['F1_all'][i] - records[type_p][proj]['F1_all'][i]]
                                #records[type_p][proj]['Time'][i]]
                    for j, m in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
                                    ['Prec', 'Pd', 'Pf', 'F1', 'G1', 'AUC_Prec', 'AUC_Pd', 'AUC_Pf', 'AUC_F1', 'AUC_G1', 'Prec_all', 'F1_all']):
                        if records[type_p][proj][m][i] != 0:
                            little_c[j] = (float(little_c[j]) * 100) / (records[type_p][proj][m][i])
                        else:
                            little_c[j] = little_c[j]
                    print(  # i, i+1,
                        int(little_c[0]), int(little_c[1]), int(little_c[2]), int(little_c[3]), int(little_c[4]),
                        int(little_c[5]), int(little_c[6]), int(little_c[7]), int(little_c[8]), int(little_c[9]),
                        little_c[10], little_c[11], int(little_c[12]), int(little_c[13]),
                        #round(little_c[9], 3),
                    sep = ",\t")
            else:
                for i in range(no_datasets - 1):
                    print(records[type_p][proj]['Prec'][i],
                          records[type_p][proj]['Pd'][i], records[type_p][proj]['Pf'][i],
                          records[type_p][proj]['F1'][i], records[type_p][proj]['G1'][i],
                          records[type_p][proj]['AUC_Prec'][i], records[type_p][proj]['AUC_Pd'][i], records[type_p][proj]['AUC_Pf'][i],
                          records[type_p][proj]['AUC_F1'][i], records[type_p][proj]['AUC_G1'][i],
                          records[type_p][proj]['IFA'][i], records[type_p][proj]['PCI20'][i], records[type_p][proj]['Prec_all'][i],
                          records[type_p][proj]['F1_all'][i],
                          #round(records[type_p][proj]['Time'][i],3),
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


if __name__ == "__main__":
    file_name = sys.argv[1]
    file_type = sys.argv[2]
    all = int(sys.argv[3])
    level = file_name.split("_")[0]
    learner = file_name.split("_")[1]
    oversampling = file_name.split("_")[2]
    if all == 1:
        file_name += "_all.p"
    else:
        file_name += "_incremental.p"
    print(level, learner, oversampling, file_type, all)
    execute(file_name, file_type, level, learner, oversampling, all)
