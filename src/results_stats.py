
import numpy as np
import pickle
import math
import sys

#projects =['mdanalysis', 'libmesh', 'lammps', 'abinit', 'hoomd', 'amber', 'pcmsolver', 'rmg-py', 'xenon']
projects =['libmesh', 'mdanalysis', 'abinit', 'lammps']

#projects =['hoomd']

data_collections = ['fastread_jit_file', 'keyword_jit_file', 'human_jit_file', 'release_level']


def reading_stats(filename, file_type):
    filehandler = open(filename, "rb")
    records = pickle.load(filehandler)
    for proj in records.keys():
        print(proj.upper())
        for type_p in records[proj].keys():
            no_datasets = len(records[type_p][proj]['Prec'])
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                "AUC_Pr", "AUC_Pd", "AUC_Pf", "AUC_F1", "AUC_G1",
                "IFA", "PCI20", "Pr_all", "F1_all",
                # "Time",
                sep=",\t")
            if type_p != file_type:
                for i in range(no_datasets):
                    little_c = [records[proj][file_type]['Prec'][i] - records[proj][type_p]['Prec'][i],
                                records[proj][file_type]['Pd'][i] - records[proj][type_p]['Pd'][i],
                                records[proj][file_type]['Pf'][i] - records[proj][type_p]['Pf'][i],
                                records[proj][file_type]['F1'][i] - records[proj][type_p]['F1'][i],
                                records[proj][file_type]['G1'][i] - records[proj][type_p]['G1'][i],
                                records[proj][file_type]['AUC_Prec'][i] - records[proj][type_p]['AUC_Prec'][i],
                                records[proj][file_type]['AUC_Pd'][i] - records[proj][type_p]['AUC_Pd'][i],
                                records[proj][file_type]['AUC_Pf'][i] - records[proj][type_p]['AUC_Pf'][i],
                                records[proj][file_type]['AUC_F1'][i] - records[proj][type_p]['AUC_F1'][i],
                                records[proj][file_type]['AUC_G1'][i] - records[proj][type_p]['AUC_G1'][i],
                                records[proj][file_type]['IFA'][i] - records[proj][type_p]['IFA'][i],
                                records[proj][file_type]['PCI20'][i] - records[proj][type_p]['PCI20'][i],
                                records[proj][file_type]['Prec_all'][i] - records[proj][type_p]['Prec_all'][i],
                                records[proj][file_type]['F1_all'][i] - records[proj][type_p]['F1_all'][i]]
                    # records[proj][type_p]['Time'][i]]
                    for j, m in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
                                    ['Prec', 'Pd', 'Pf', 'F1', 'G1', 'AUC_Prec', 'AUC_Pd', 'AUC_Pf', 'AUC_F1', 'AUC_G1',
                                     'Prec_all', 'F1_all']):
                        if records[proj][type_p][m][i] != 0:
                            #little_c[j] = (float(little_c[j]) * 100) / (records[proj][type_p][m][i])
                            little_c[j] = little_c[j]
                        else:
                            little_c[j] = little_c[j]
                    print(  # i, i+1,
                        int(little_c[0]), int(little_c[1]), int(little_c[2]), int(little_c[3]), int(little_c[4]),
                        int(little_c[5]), int(little_c[6]), int(little_c[7]), int(little_c[8]), int(little_c[9]),
                        little_c[10], little_c[11], int(little_c[12]), int(little_c[13]),
                        # round(little_c[9], 3),
                        sep=",\t")
            else:
                for i in range(no_datasets - 1):
                    print(records[proj][type_p]['Prec'][i],
                          records[proj][type_p]['Pd'][i], records[proj][type_p]['Pf'][i],
                          records[proj][type_p]['F1'][i], records[proj][type_p]['G1'][i],
                          records[proj][type_p]['AUC_Prec'][i], records[proj][type_p]['AUC_Pd'][i],
                          records[proj][type_p]['AUC_Pf'][i],
                          records[proj][type_p]['AUC_F1'][i], records[proj][type_p]['AUC_G1'][i],
                          records[proj][type_p]['IFA'][i], records[proj][type_p]['PCI20'][i],
                          records[proj][type_p]['Prec_all'][i],
                          records[proj][type_p]['F1_all'][i],
                          # round(records[proj][type_p]['Time'][i],3),
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


def reading_stats_all(filename):
    filehandler = open(filename, "rb")
    records = pickle.load(filehandler)
    for proj in projects:
        print(proj.upper())
        for type_p in records[proj].keys():
            no_datasets = len(records[proj][type_p]['Prec'])
            print(type_p.upper(), no_datasets + 1)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                "AUC_Pr", "AUC_Pd", "AUC_Pf", "AUC_F1", "AUC_G1",
                "IFA", "PCI20", "Pr_all", "F1_all",
                # "Time",
                sep=",\t")
            #print("hehehe")
            for i in range(no_datasets):
                print(records[proj][type_p]['Prec'][i],
                      records[proj][type_p]['Pd'][i], records[proj][type_p]['Pf'][i],
                      records[proj][type_p]['F1'][i], records[proj][type_p]['G1'][i],
                      records[proj][type_p]['AUC_Prec'][i], records[proj][type_p]['AUC_Pd'][i],
                      records[proj][type_p]['AUC_Pf'][i],
                      records[proj][type_p]['AUC_F1'][i], records[proj][type_p]['AUC_G1'][i],
                      records[proj][type_p]['IFA'][i], records[proj][type_p]['PCI20'][i],
                      records[proj][type_p]['Prec_all'][i],
                      records[proj][type_p]['F1_all'][i],
                      # round(records[proj][type_p]['Time'][i],3),
                      sep=",\t")
            print("\n" + 50 * "-" + "\n")


def reading_stats_fft_all(file_name):
    filehandler = open(file_name, "rb")
    records = pickle.load(filehandler)
    for proj in projects:
        print(proj.upper())
        for type_p in records[proj].keys():
            no_datasets = len(records[proj][type_p]['Prec'])
            print(type_p.upper(), no_datasets + 1)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                # "Time",
                sep=",\t")
            # print("hehehe")
            for i in range(no_datasets):
                print(records[proj][type_p]['Prec'][i],
                      records[proj][type_p]['Pd'][i], records[proj][type_p]['Pf'][i],
                      records[proj][type_p]['F1'][i], records[proj][type_p]['G1'][i],
                      # round(records[proj][type_p]['Time'][i],3),
                      sep=",\t")
            print("\n" + 50 * "-" + "\n")


def reading_stats_fft(file_name, file_type):
    filehandler = open(file_name, "rb")
    records = pickle.load(filehandler)
    for proj in records.keys():
        print(proj)
        for type_p in records[proj].keys():
            print(type_p)
            dataset = records[proj][type_p]['Prec']
            no_datasets = len(dataset)
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf", "F1", "G1",
                "AUC_Pr", "AUC_Pd", "AUC_Pf", "AUC_F1", "AUC_G1",
                "IFA", "PCI20",
                # "Time",
                sep=",\t")
            if type_p != file_type:
                for i in range(no_datasets):
                    little_c = [records[proj][file_type]['Prec'][i] - records[proj][type_p]['Prec'][i],
                                records[proj][file_type]['Pd'][i] - records[proj][type_p]['Pd'][i],
                                records[proj][file_type]['Pf'][i] - records[proj][type_p]['Pf'][i],
                                records[proj][file_type]['F1'][i] - records[proj][type_p]['F1'][i],
                                records[proj][file_type]['G1'][i] - records[proj][type_p]['G1'][i]]
                    for j, m in zip([0, 1, 2, 3, 4],
                                    ['Prec', 'Pd', 'Pf', 'F1',
                                     'G1']):  # 5, 6, 7, 8, 9], #, 'AUC_Prec', 'AUC_Pd', 'AUC_Pf', 'AUC_F1', 'AUC_G1']):
                        if records[proj][type_p][m][i] != 0:
                            little_c[j] = (float(little_c[j]) * 100) / (records[proj][type_p][m][i])
                    print(int(little_c[0]), int(little_c[1]), int(little_c[2]), int(little_c[3]), int(little_c[4]),
                        sep=",\t")
            else:
                for i in range(no_datasets):
                    print(records[proj][type_p]['Prec'][i],
                          records[proj][type_p]['Pd'][i], records[proj][type_p]['Pf'][i],
                          records[proj][type_p]['F1'][i], records[proj][type_p]['G1'][i],
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


def read_metrics_full(metric):
    file_names = ['commit_LR_nosmote_0_incremental.p', 'commit_LR_amrit_0_incremental.p', 'commit_LR_sk_0_incremental.p',
                  'commit_RF_nosmote_0_incremental.p', 'commit_RF_amrit_0_incremental.p', 'commit_RF_sk_0_incremental.p',
                  'commit_LinearSVC_nosmote_0_incremental.p', 'commit_LinearSVC_amrit_0_incremental.p', 'commit_LinearSVC_sk_0_incremental.p']
                  #'commit_FFT_amrit_G1_0_temp_incremental.p', 'commit_FFT_nosmote_G1_0_temp_incremental.p']
                  #'commit_FFT_nosmote_G1_0_full_incremental.p', 'commit_FFT_amrit_G1_0_full_incremental.p', 'commit_FFT_sk_G1_0_full_incremental.p']
    records = []
    index = 0
    for f in file_names:
        print(index)
        index += 1
        records.append(pickle.load(open(f, "rb")))

    for proj in projects:
        print(proj.upper())
        for type_p in records[0][proj].keys():
            no_datasets = len(records[0][proj][type_p]['Prec'])
            print(type_p.upper(), no_datasets + 1)
            print("LR", "RF", "SVM", "FFT_G1", "FFT_D2H",
                # "Time",
                sep=",\t")
            for i in range(no_datasets):
                index = 0
                for rec in records:
                    if index < 9 and metric == "G1":
                        print(rec[proj][type_p][metric + "_all"][i],
                            end=", ")
                    else:
                        print(rec[proj][type_p][metric][i],
                            end=", ")
                    index += 1
                print()
            print("\n" + 50 * "-" + "\n")


def read_metrics_partial(metric):
    file_names = ['commit_LR_nosmote_0_temp_incremental.p', 'commit_LR_amrit_0_temp_incremental.p', 'commit_LR_sk_0_temp_incremental.p',
                  'commit_RF_nosmote_0_temp_incremental.p', 'commit_RF_amrit_0_temp_incremental.p', 'commit_RF_sk_0_temp_incremental.p',
                  'commit_LinearSVC_nosmote_0_temp_incremental.p', 'commit_LinearSVC_amrit_0_temp_incremental.p', 'commit_LinearSVC_sk_0_temp_incremental.p',
                  'commit_FFT_nosmote_G1_0_partial_incremental.p', 'commit_FFT_amrit_G1_0_partial_incremental.p', 'commit_FFT_sk_G1_0_partial_incremental.p']
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))

    if metric == "IFA":
        records = records[:9]

    for proj in projects:
        print(proj.upper())
        for type_p in records[0][proj].keys():
            no_datasets = len(records[0][proj][type_p]['Prec'])
            print(type_p.upper(), no_datasets + 1)
            print("LR", "RF", "SVM", "FFT_G1", "FFT_D2H",
                # "Time",
                sep=",\t")
            for i in range(no_datasets):
                index = 0
                for rec in records:
                    print(rec[proj][type_p][metric][i],
                            end=", ")
                    index += 1
                print()
            print("\n" + 50 * "-" + "\n")


def rq1_second(metric):
    file_names = ['commit_LR_amrit_0_incremental.p', 'commit_RF_amrit_0_incremental.p',
                  'commit_LinearSVC_amrit_0_incremental.p', 'commit_FFT_amrit_G1_0_full_incremental.p']
    #file_names = ['commit_LR_nosmote_0_temp_incremental.p', 'commit_RF_nosmote_0_temp_incremental.p',
    #              'commit_LinearSVC_nosmote_0_temp_incremental.p', 'commit_FFT_nosmote_G1_0_full_incremental.p']
    #file_names = ['commit_LR_sk_0_temp_incremental.p', 'commit_RF_sk_0_temp_incremental.p',
    #              'commit_LinearSVC_sk_0_temp_incremental.p', 'commit_FFT_sk_G1_0_full_incremental.p']
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))

    for proj in projects:
        no_datasets = len(records[0][proj]['fastread']['Prec'])
        print(proj.upper(), no_datasets + 1)
        index = 0
        for rec in records:
            if index < 3:
                print(file_names[index].split("_")[1],
                      hedges(rec[proj]['keyword'][metric + "_all"], rec[proj]['fastread'][metric + "_all"]))
            else:
                print(file_names[index].split("_")[1],
                      hedges(rec[proj]['keyword'][metric], rec[proj]['fastread'][metric]))
            index += 1
        print("\n" + 50 * "-" + "\n")


def rq1_human(metric):
    file_names = ['commit_LR_amrit_0_human_incremental.p', 'commit_RF_amrit_0_human_incremental.p',
                  'commit_LinearSVC_amrit_0_human_incremental.p', 'commit_FFT_amrit_G1_0_human_incremental.p']
    #file_names = ['commit_LR_nosmote_0_temp_incremental.p', 'commit_RF_nosmote_0_temp_incremental.p',
    #              'commit_LinearSVC_nosmote_0_temp_incremental.p', 'commit_FFT_nosmote_G1_0_full_incremental.p']
    #file_names = ['commit_LR_sk_0_temp_incremental.p', 'commit_RF_sk_0_temp_incremental.p',
    #              'commit_LinearSVC_sk_0_temp_incremental.p', 'commit_FFT_sk_G1_0_full_incremental.p']
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))

    for proj in projects:
        no_datasets = len(records[0][proj]['fastread']['Prec'])
        print(proj.upper(), no_datasets + 1)
        index = 0
        for rec in records:
            if index < 3:
                print(file_names[index].split("_")[1],
                      hedges(rec[proj]['keyword'][metric + "_all"], rec[proj]['fastread'][metric + "_all"]))
            else:
                print(file_names[index].split("_")[1],
                      hedges(rec[proj]['keyword'][metric], rec[proj]['fastread'][metric]))
            index += 1
        print("\n" + 50 * "-" + "\n")


def hedges(i, j, small=0.38):
    """
    Hedges effect size test.
    Returns true if the "i" and "j" difference is only a small effect.
    "i" and "j" are   objects reporing mean (i.mu), standard deviation (i.s)
    and size (i.n) of two  population of numbers.
    """
    num   = (len(i) - 1)*(np.std(i)**2) + (len(j) - 1)*(np.std(j)**2)
    denom = (len(i) - 1) + (len(j) - 1)
    sp    = (num / denom)**0.5
    delta = abs(np.mean(i) - np.mean(j)) / sp
    c     = 1 - 3.0 / (4*(len(i) + len(j) - 2) - 1)
    #print(num, denom, sp, delta, c)
    return delta * c < small


if __name__ == "__main__":
    file_name = sys.argv[1]
    file_type = sys.argv[2]
    all_type = sys.argv[3]
    print(file_name, file_type, all_type)
    learner = file_name.split("_")[1]
    if learner != "FFT":
        if int(all_type) == 1:
            reading_stats_all(file_name)
        else:
            reading_stats(file_name, file_type)
    else:
        if int(all_type) == 1:
            reading_stats_fft_all(file_name)
        else:
            reading_stats_fft(file_name, file_type)

    #read_metrics_full('G1')
    #rq1_human("G1")

