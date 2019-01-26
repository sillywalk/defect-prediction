from __future__ import print_function, division
__author__ = 'huy'
"""
A Defect Prediction model for file level metrics
"""
import os
import re
import random
import sys
sys.path.append('/home/huyqt7/Projects/PhD/defect-prediction/src/fft_src/')
import numpy as np
import pandas as pd
import pdb
import warnings
warnings.filterwarnings("ignore")
import pickle
import time
import random



from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
print(root)
if root not in sys.path:
    sys.path.append(str(root))
fft_root = os.path.join(str(root), "fft_src")
print(fft_root)


from metrics.abcd import ABCD
from data.data_handler import DataHandler
from prediction.model import PredictionModel

import sys


random.seed(9001)
np.random.seed(47)
projects =['pcmsolver', 'rmg-py', 'amber', 'mdanalysis', 'libmesh', 'lammps', 'abinit', 'hoomd', 'xenon']
#projects =['rmg-py']
data_collections = ['fastread_jit_file', 'release_level', 'keyword_jit_file', 'human_jit_file']


def execute(filename, file_type, level, learner, sampling, learning_goal, reduce, partial, all):
    dh = DataHandler(level)
    data = dh.get_data()
    #set_trace()
    records = {}

    for proj in projects:
        print(proj.upper())
        for type_p in data[proj].keys():
            if proj not in records.keys():
                records[proj] = {}
            records[proj][type_p] = {"Prec": [], "Pd": [], "Pf": [], "F1":[],  "G1": [],
                                     "AUC_Prec": [], "AUC_Pd": [], "AUC_Pf": [], "AUC_F1": [], "AUC_G1": [],
                                     "IFA":[], "PCI20":[],
                "Prec_all":[], "F1_all":[], "Time": []}
            dataset = data[proj][type_p]

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
                    if "_jit_file" not in file_type:
                        file_type += "_jit_file"
                    test_dataframe = data[proj]['human_jit_file'][i + 1]
                    # Get lines of code of the test instances
                    loc = test_dataframe["CountLine"].values.tolist()
                    if type_p == "release_level":
                        train_dataframe = train_dataframe.drop(['fastread_bugs', 'human_bugs'], axis=1)
                else:
                    #if proj in ['mdanalysis', 'libmesh', 'lammps', 'abinit']:
                    #    test_dataframe = data[proj]['human'][i + 1]
                    #else:
                    test_dataframe = dataset[i + 1]
                    loc = test_dataframe["lt"] * test_dataframe["nf"]
                    loc = loc.tolist()

                p_d, p_f = [], []
                f_1, prec = [], []
                g_1 = []
                #ifa , pci20 = [], []
                #auc_pd, auc_pf, auc_prec, auc_f1, auc_g1 = [], [], [], [], []
                #total_prec, total_f1 = [], []

                running_times = []
                pred_mod = PredictionModel(classifier=learner)
                for _ in range(10):
                    # Make predictions
                    start_time = time.time()
                    if reduce == "1":
                        #print("yes reduce")
                        n_train_samples = int(train_dataframe.shape[0] * .5)
                        #n_test_samples = int(test_dataframe.shape[0] * 1)
                        train_sample_indices = random.sample(range(0, train_dataframe.shape[0]), n_train_samples)
                        #test_sample_indices = random.sample(range(0, test_dataframe.shape[0]), n_test_samples)
                        #test_sample_indices.sort()
                        train_sample_indices.sort()
                        #print("train_df", train_dataframe.shape)
                        temp_train_df = train_dataframe.iloc[train_sample_indices]
                        #temp_test_df = test_dataframe.iloc[test_sample_indices]

                        try:
                            TP, FP, TN, FN = pred_mod.predict_defects(temp_train_df, test_dataframe,
                                                                      samplingtechnique=sampling, loc=loc,
                                                                      goal=learning_goal, partial=partial)
                        except ValueError:
                            break
                    else:
                        #print("no reduce")
                        try:
                            TP, FP, TN, FN = pred_mod.predict_defects(train_dataframe, test_dataframe,
                                                                      samplingtechnique=sampling, loc=loc,
                                                                      goal=learning_goal, partial=partial)
                        except ValueError:
                            break

                    #import pdb
                    #pdb.set_trace()
                    delta_time = time.time() - start_time
                    abcd = ABCD(fft_results=[TP, FP, TN, FN])

                    # Get performance metrics
                    #ifa_obtained = abcd.get_ifa()
                    #pci20_obtained = abcd.get_pci_20()
                    pd_obtained, pf_obtained = abcd.get_pd_pf()
                    prec_obtained, f1_obtained = abcd.get_f_score()
                    g1_obtained = abcd.get_G()
                    #auc_pd_val, auc_pf_val, auc_prec_val, auc_f1_val, auc_g1_val = abcd._set_auc(3)
                    #total_prec_val, total_f1_val = abcd.get_score_total()
                    # Gather the obtained performance metrics
                    prec.append(prec_obtained)
                    p_d.append(pd_obtained)
                    p_f.append(pf_obtained)
                    f_1.append(f1_obtained)
                    g_1.append(g1_obtained)
                    #auc_pd.append(auc_pd_val)
                    #auc_pf.append(auc_pf_val)
                    #auc_prec.append(auc_prec_val)
                    #auc_f1.append(auc_f1_val)
                    #auc_g1.append(auc_g1_val)
                    #ifa.append(ifa_obtained)
                    #pci20.append(pci20_obtained)
                    #total_prec.append(total_prec_val)
                    #total_f1.append(total_f1_val)
                    running_times.append(delta_time)
                    #print(g1_obtained)

                i += 1
                little_res = [np.median(prec).astype("int"),
                              np.median(p_d).astype("int"),
                              np.median(p_f).astype("int"),
                              np.median(f_1).astype("int"),
                              np.median(g_1).astype("int")]
                              #np.median(auc_prec).astype("int"),
                              #np.median(auc_pd).astype("int"),
                              #np.median(auc_pf).astype("int"),
                              #np.median(auc_f1).astype("int"),
                              #np.median(auc_g1).astype("int")]
                              #np.median(ifa).astype("int"),
                              #np.median(pci20).astype("int")]
                              #np.median(total_prec).astype("int"),
                              #np.median(total_f1).astype("int")]
                              #np.median(running_times)]
                records[proj][type_p]['Prec'].append(little_res[0])
                records[proj][type_p]['Pd'].append(little_res[1])
                records[proj][type_p]['Pf'].append(little_res[2])
                records[proj][type_p]['F1'].append(little_res[3])
                records[proj][type_p]['G1'].append(little_res[4])
                #records[proj][type_p]['AUC_Prec'].append(little_res[5])
                #records[proj][type_p]['AUC_Pd'].append(little_res[6])
                #records[proj][type_p]['AUC_Pf'].append(little_res[7])
                #records[proj][type_p]['AUC_F1'].append(little_res[8])
                #records[proj][type_p]['AUC_G1'].append(little_res[9])
                #records[proj][type_p]['IFA'].append(little_res[10])
                #records[proj][type_p]['PCI20'].append(little_res[11])
                #records[proj][type_p]['Prec_all'].append(little_res[12])
                #records[proj][type_p]['F1_all'].append(little_res[13])
                #records[proj][type_p]['Time'].append(little_res[9])



                #print(#i, i+1,
                #      little_res,
                #      sep=",\t")
            print()
            #set_trace()
        print("save + ", filename)
        filehandler = open(filename, "wb")
        pickle.dump(records, filehandler)
        for type_p in data[proj].keys():
            print(type_p)
            dataset = data[proj][type_p]
            no_datasets = len(dataset)
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                "AUC_Pr", "AUC_Pd", "AUC_Pf", "AUC_F1", "AUC_G1",
                "IFA", "PCI20",
                #"Time",
                sep=",\t")
            if type_p != file_type:
                for i in range(no_datasets - 1):
                    little_c = [records[proj][file_type]['Prec'][i] - records[proj][type_p]['Prec'][i],
                                records[proj][file_type]['Pd'][i] - records[proj][type_p]['Pd'][i],
                                records[proj][file_type]['Pf'][i] - records[proj][type_p]['Pf'][i],
                                records[proj][file_type]['F1'][i] - records[proj][type_p]['F1'][i],
                                records[proj][file_type]['G1'][i] - records[proj][type_p]['G1'][i]]
                                #records[proj][file_type]['AUC_Prec'][i] - records[proj][type_p]['AUC_Prec'][i],
                                #records[proj][file_type]['AUC_Pd'][i] - records[proj][type_p]['AUC_Pd'][i],
                                #records[proj][file_type]['AUC_Pf'][i] - records[proj][type_p]['AUC_Pf'][i],
                                #records[proj][file_type]['AUC_F1'][i] - records[proj][type_p]['AUC_F1'][i],
                                #records[proj][file_type]['AUC_G1'][i] - records[proj][type_p]['AUC_G1'][i],
                                #records[proj][file_type]['IFA'][i] - records[proj][type_p]['IFA'][i],
                                #records[proj][file_type]['PCI20'][i] - records[proj][type_p]['PCI20'][i]]
                                #records[proj][file_type]['Prec_all'][i] - records[proj][type_p]['Prec_all'][i],
                                #records[proj][file_type]['F1_all'][i] - records[proj][type_p]['F1_all'][i]]
                                #records[proj][type_p]['Time'][i]]
                    for j, m in zip([0, 1, 2, 3, 4],
                                    ['Prec', 'Pd', 'Pf', 'F1', 'G1']): #5, 6, 7, 8, 9], #, 'AUC_Prec', 'AUC_Pd', 'AUC_Pf', 'AUC_F1', 'AUC_G1']):
                        if records[proj][type_p][m][i] != 0:
                            little_c[j] = (float(little_c[j]) * 100) / (records[proj][type_p][m][i])
                    print(  # i, i+1,
                        int(little_c[0]), int(little_c[1]), int(little_c[2]), int(little_c[3]), int(little_c[4]),
                        #int(little_c[5]), int(little_c[6]), int(little_c[7]), int(little_c[8]), int(little_c[9]),
                        #little_c[10], little_c[11], int(little_c[12]), int(little_c[13]),
                        #round(little_c[9], 3),
                    sep = ",\t")
            else:
                for i in range(no_datasets - 1):
                    print(records[proj][type_p]['Prec'][i],
                          records[proj][type_p]['Pd'][i], records[proj][type_p]['Pf'][i],
                          records[proj][type_p]['F1'][i], records[proj][type_p]['G1'][i],
                          #records[proj][type_p]['AUC_Prec'][i], records[proj][type_p]['AUC_Pd'][i], records[proj][type_p]['AUC_Pf'][i],
                          #records[proj][type_p]['AUC_F1'][i], records[proj][type_p]['AUC_G1'][i],
                          #records[proj][type_p]['IFA'][i], records[proj][type_p]['PCI20'][i],
                          #records[proj][type_p]['Prec_all'][i],records[proj][type_p]['F1_all'][i],
                          #round(records[proj][type_p]['Time'][i],3),
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


if __name__ == "__main__":
    file_name = sys.argv[1]
    file_type = sys.argv[2]
    all = int(sys.argv[3])
    level = file_name.split("_")[0]
    learner = file_name.split("_")[1]
    oversampling = file_name.split("_")[2]
    learning_goal = file_name.split("_")[3]
    reduce = file_name.split("_")[4]
    partial = file_name.split("_")[5]

    if all == 1:
        file_name += "_all.p"
    else:
        file_name += "_incremental.p"

    if partial == "partial":
        p_condition = True
    else:
        p_condition = False
    print(level, learner, oversampling, file_type, learning_goal, reduce, p_condition, all)
    execute(file_name, file_type, level, learner, oversampling, learning_goal, reduce, p_condition, all)
