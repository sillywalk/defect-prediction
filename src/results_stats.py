
import numpy as np
import pickle
import sys

projects =['mdanalysis', 'lammps', 'libmesh', 'abinit']
data_collections = ['fastread_jit_file', 'keyword_jit_file', 'human_jit_file', 'release_level']


def reading_stats(filename, file_type):
    filehandler = open(filename, "rb")
    records = pickle.load(filehandler)
    for proj in projects:
        print(proj.upper())
        for type_p in data_collections:
            no_datasets = len(records[type_p][proj]['Prec'])
            print(type_p.upper(), no_datasets)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf",
                "F1", "IFA", "PCI20",
                "Prec_all", "F1_all", "Time",
                sep=",\t")
            if type_p != file_type:
                for i in range(no_datasets):
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
                        sep=",\t")
            else:
                for i in range(no_datasets - 1):
                    print(records[type_p][proj]['Prec'][i],
                          records[type_p][proj]['Pd'][i], records[type_p][proj]['Pf'][i],
                          records[type_p][proj]['F1'][i], records[type_p][proj]['IFA'][i],
                          records[type_p][proj]['PCI20'][i], records[type_p][proj]['Prec_all'][i],
                          records[type_p][proj]['F1_all'][i], round(records[type_p][proj]['Time'][i], 3),
                          sep=",\t")
            print("\n" + 50 * "-" + "\n")


def reading_stats_all(filename):
    filehandler = open(filename, "rb")
    records = pickle.load(filehandler)
    for proj in projects:
        print(proj.upper())
        for type_p in data_collections:
            no_datasets = len(records[type_p][proj]['Prec'])
            print(type_p.upper(), no_datasets + 1)
            print(  # "Train", "Test",
                "Prec", "Pd", "Pf",
                "F1", "IFA", "PCI20",
                "Prec_all", "F1_all", "Time",
                sep=",\t")
            #print("hehehe")
            for i in range(no_datasets):
                print(records[type_p][proj]['Prec'][i],
                      records[type_p][proj]['Pd'][i], records[type_p][proj]['Pf'][i],
                      records[type_p][proj]['F1'][i], records[type_p][proj]['IFA'][i],
                      records[type_p][proj]['PCI20'][i], records[type_p][proj]['Prec_all'][i],
                      records[type_p][proj]['F1_all'][i], round(records[type_p][proj]['Time'][i], 3),
                      sep=",\t")
            print("\n" + 50 * "-" + "\n")



if __name__ == "__main__":
    file_name = sys.argv[1]
    file_type = sys.argv[2]
    all_type = sys.argv[3]
    print(file_name, file_type, all_type)
    if int(all_type) == 1:
        reading_stats_all(file_name)
    else:
        reading_stats(file_name, file_type)