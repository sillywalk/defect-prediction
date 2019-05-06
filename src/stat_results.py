from __future__ import division, print_function
import sys, random, argparse
sys.path.append('/src')
print(sys.path)
from stat_helpers import *

def rq4_cgstats(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_nosmote_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    for proj in projects:
        collect[proj] = {"FFFT": 0, "KLR": 0}
        for index in range(len(records[0][proj])):
            one_record = {"FFFT": records[1][proj][index]['fastread'][perf_measure],
                          "KLR": records[0][proj][index]['keyword'][perf_measure + "_all"]}

            ranks = rdivDemo1([[k] + v for k, v in one_record.items()])
            rs = ranks.values()
            rs_counter = collections.Counter(rs)
            if rs_counter[1] == 1:
                for key, rank in ranks.items():
                    if rank == 1:
                        collect[proj][key] += 1
    with open("stats_rq4_cgstats_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)
    return collect


def rq4_cgdiff(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_nosmote_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    print()
    print(perf_measure)
    print("Dataset \t| absolute | relative")
    for proj in projects:
        collect[proj] = {'absolute': [], 'relative': []}
        for index in range(len(records[0][proj])):
            f3t = np.median(records[1][proj][index]['fastread'][perf_measure])
            cguru = np.median(records[0][proj][index]['keyword'][perf_measure + "_all"])
            delta =  f3t - cguru
            if cguru > 0:
                delta_per_p = [delta, (delta * 100.0) / cguru]
            else:
                delta_per_p = [delta, 100]
            collect[proj]['absolute'].append(delta_per_p[0])
            collect[proj]['relative'].append(delta_per_p[1])

        if len(proj.upper()) > 8:
            str_proj = proj.upper() + "\t,    %s\t   ,    %s" % (
            int(np.percentile(collect[proj]['absolute'], 50)), int(np.percentile(collect[proj]['relative'], 50)))
        else:
            str_proj = proj.upper() + "\t\t,    %s\t   ,    %s" % (
            int(np.percentile(collect[proj]['absolute'], 50)), int(np.percentile(collect[proj]['relative'], 50)))

        print(str_proj)
    with open("stats_rq4_cgdiff_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)
    return collect


def rq4_stats(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_sk_0_human_incremental.p', 'sk_commit_RF_sk_0_human_incremental.p',
                      'sk_commit_LinearSVC_sk_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    for proj in projects:
        collect[proj] = {"F3T": 0, "K+S+LR": 0, "K+S+RF": 0, "K+S+SVM": 0, 'total': len(records[0][proj])}
        for index in range(len(records[0][proj])):
            one_record = {"F3T": records[3][proj][index]['fastread'][perf_measure],
                          "K+S+LR": records[0][proj][index]['keyword'][perf_measure + "_all"],
                          "K+S+RF": records[1][proj][index]['keyword'][perf_measure + "_all"],
                          "K+S+SVM": records[2][proj][index]['keyword'][perf_measure + "_all"]}

            ranks = rdivDemo1([[k] + v for k, v in one_record.items()])
            rs = ranks.values()
            rs_counter = collections.Counter(rs)
            if rs_counter[1] == 1:
                for key, rank in ranks.items():
                    if rank == 1:
                        collect[proj][key] += 1
    with open("stats_rq4_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)

    return collect

def rq4_diff(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_sk_0_human_incremental.p', 'sk_commit_RF_sk_0_human_incremental.p',
                    'sk_commit_LinearSVC_sk_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    print()
    print(perf_measure)
    print(" "*25, "ABSOLUTE DIFF", " "*35, "RELATIVE DIFF")
    print("Dataset \t|   SVM    |    LR      |    RF   \t\tSVM\t   |    LR      |    RF")
    for proj in projects:
        collect[proj] = {}
        collect[proj]['LR'] = {'absolute': [], 'relative': []}
        collect[proj]['RF'] = {'absolute': [], 'relative': []}
        collect[proj]['SVM'] = {'absolute': [], 'relative': []}

        for index in range(len(records[0][proj])):
            delta_per_p = [0, 0]
            one_record = {"LR": np.median(records[0][proj][index]['keyword'][perf_measure + "_all"]),
                        "RF": np.median(records[1][proj][index]['keyword'][perf_measure + "_all"]),
                        "SVM": np.median(records[2][proj][index]['keyword'][perf_measure + "_all"])}
            for learner in one_record.keys():
                delta = np.median(records[3][proj][index]['fastread'][perf_measure]) - one_record[learner]

                if not math.isnan(one_record[learner]):
                    if one_record[learner] > 0:
                        delta_per_p = [delta, (delta * 100.0) / one_record[learner]]
                    else:
                        delta_per_p = [delta, 100]
                collect[proj][learner]['absolute'].append(delta_per_p[0])
                collect[proj][learner]['relative'].append(delta_per_p[1])
        entries = [proj.upper(), int(np.percentile(collect[proj]['SVM']['absolute'], 50)),
                   int(np.percentile(collect[proj]['LR']['absolute'], 50)), int(np.percentile(collect[proj]['RF']['absolute'], 50)),
                   int(np.percentile(collect[proj]['SVM']['relative'], 50)),
                   int(np.percentile(collect[proj]['LR']['relative'], 50)), int(np.percentile(collect[proj]['RF']['relative'], 50))]
        #print(len(entries))
        str_proj = str(entries[0]) + "\t"
        if len(proj.upper()) < 9:
            str_proj += "\t"
        str_proj += ",    %s\t   ,     %s\t,    %s\t\t\t %s\t   ,     %s\t,    %s" % \
                    (entries[1], entries[2], entries[3], entries[4], entries[5], entries[6])

        print(str_proj)
    with open("stats_rq4_diff_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)
    print()
    return collect


def rq2_stats(perf_measure, projects, filename):
    records = []
    file_names = [filename]
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    for proj in projects:
        collect[proj] = {"fastread": 0, "keyword": 0, 'total': len(records[0][proj])}
        for index in range(len(records[0][proj])):
            one_record = {"fastread": records[0][proj][index]['fastread'][perf_measure],
                          "keyword": records[0][proj][index]['keyword'][perf_measure]}

            ranks = rdivDemo1([[k] + v for k, v in one_record.items()])
            rs = ranks.values()
            rs_counter = collections.Counter(rs)

            if rs_counter[1] == 1:
                for key, rank in ranks.items():
                    if rank == 1:
                        collect[proj][key] += 1
    with open("stats_rq2_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)
    return collect

def rq2_diff(perf_measure, projects, filename):
    records = []
    file_names = [filename]
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    delta_per_p = [0, 0]
    print("\n" + perf_measure)
    print("Dataset \t| absolute | relative")
    for proj in projects:
        collect[proj] = {'absolute': [], 'relative': []}
        for index in range(len(records[0][proj])):
            delta = np.median(records[0][proj][index]['fastread'][perf_measure]) - \
                        np.median(records[0][proj][index]['keyword'][perf_measure])

            if math.isnan(delta):
                collect[proj].append(0)
            else:
                keyword_val = np.median(records[0][proj][index]['keyword'][perf_measure])
                if not math.isnan(keyword_val):
                    if keyword_val > 0:
                        delta_per_p = [delta, (delta * 100.0) / keyword_val]
                    else:
                        delta_per_p = [delta, 100]
                collect[proj]['absolute'].append(delta_per_p[0])
                collect[proj]['relative'].append(delta_per_p[1])
        if len(proj.upper()) > 8:
            str_proj = proj.upper() + "\t,    %s     ,    %s" % (int(np.percentile(collect[proj]['absolute'], 50)), int(np.percentile(collect[proj]['relative'], 50)))
        else:
            str_proj = proj.upper() + "\t\t,    %s     ,    %s" % (int(np.percentile(collect[proj]['absolute'], 50)), int(np.percentile(collect[proj]['relative'], 50)))
        print(str_proj)

    with open("stats_rq2_diff.p", 'wb') as handle:
        pickle.dump(collect, handle)
    #print("finish rq5")
    return collect



def rq3_stats(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_sk_0_human_incremental.p', 'sk_commit_RF_sk_0_human_incremental.p',
                      'sk_commit_LinearSVC_sk_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    for proj in projects:
        collect[proj] = {"FFT": 0, "LR": 0, "RF": 0, "SVM": 0, 'total': len(records[0][proj])}
        for index in range(len(records[0][proj])):
            one_record = {"FFT": records[3][proj][index]['fastread'][perf_measure],
                          "LR": records[0][proj][index]['fastread'][perf_measure + "_all"],
                          "RF": records[1][proj][index]['fastread'][perf_measure + "_all"],
                          "SVM": records[2][proj][index]['fastread'][perf_measure + "_all"]}

            ranks = rdivDemo1([[k] + v for k, v in one_record.items()])
            rs = ranks.values()
            rs_counter = collections.Counter(rs)
            if rs_counter[1] == 1:
                for key, rank in ranks.items():
                    if rank == 1:
                        collect[proj][key] += 1
    with open("stats_rq3_%s.p" % perf_measure, 'wb') as handle:
        pickle.dump(collect, handle)


def rq3_diff(perf_measure, projects, filename):
    file_names = ['sk_commit_LR_sk_0_human_incremental.p', 'sk_commit_RF_sk_0_human_incremental.p',
                    'sk_commit_LinearSVC_sk_0_human_incremental.p', filename]
    records = []
    for f in file_names:
        records.append(pickle.load(open(f, "rb")))
    collect = {}
    for proj in projects:
        collect[proj] = {}
        collect[proj]['LR'] = []
        collect[proj]['RF'] = []
        collect[proj]['SVM'] = []
        for index in range(len(records[0][proj])):
            one_record = {"LR": np.median(records[0][proj][index]['fastread'][perf_measure + "_all"]),
                        "RF": np.median(records[1][proj][index]['fastread'][perf_measure + "_all"]),
                        "SVM": np.median(records[2][proj][index]['fastread'][perf_measure + "_all"])}
            for learner in one_record.keys():
                delta = np.median(records[3][proj][index]['fastread'][perf_measure]) - one_record[learner]
                if math.isnan(delta):
                    collect[proj][learner].append(0)
                else:
                    collect[proj][learner].append(delta)
    print("rq3: absolute median difference between all learners and F3T")
    for iq in [50]:
        for proj in projects:
            print(proj.upper(), int(np.percentile(collect[proj]['SVM'], iq)) ,int(np.percentile(collect[proj]['LR'], iq)),
                  int(np.percentile(collect[proj]['RF'], iq)), sep=",")
        print()
    with open("stats_rq3_diff.p", 'wb') as handle:
        pickle.dump(collect, handle)

    return collect


if __name__ == "__main__":
    import collections
    import pickle
    rq = sys.argv[1]
    projects =['mdanalysis', 'libmesh', 'lammps', 'abinit', 'hoomd', 'amber', 'pcmsolver', 'rmg-py', 'xenon']

    files_name = ["sk_commit_FFT_nosmote_G1.p", "sk_commit_FFT_nosmote_P_opt20.p"]
    perf_measures = ['G1', 'P_opt20']
    if rq == "rq2":
        print("rq2: difference between K+FFT and FASTREAD+FFTs(F3T)")
        rq2_diff(perf_measures[0], projects, files_name[0])
        print()
        collect = rq2_stats(perf_measures[0], projects, files_name[0])
        rq2_diff(perf_measures[1], projects, files_name[1])
        print()
        collect = rq2_stats(perf_measures[1], projects, files_name[1])
        header = "Dataset \t|    Keyword    |    FASTREAD"
    elif rq == "rq3":
        collect = rq3_stats(perf_measures[0], projects, files_name[0])
        collect = rq3_stats(perf_measures[1], projects, files_name[1])
        print()
        header = "Dataset \t|  SMOTE+SVM\t|   SMOTE+RF\t|    SMOTE+LR\t|\tFFT"
    else:
        print("rq4: difference of commit.guru(keyword+LR) and F3T(FASTREAD+FFTs) performance")
        rq4_cgdiff(perf_measures[0], projects, files_name[0])
        rq4_cgdiff(perf_measures[1], projects, files_name[1])
        print("\nrq4: difference of traditional system(keyword+SMOTE+learner) and F3T(FASTREAD+FFTs) performance")
        rq4_diff(perf_measures[0], projects, files_name[0])
        rq4_diff(perf_measures[1], projects, files_name[1])
        print()
        collect = rq4_stats(perf_measures[0], projects, files_name[0])
        collect = rq4_stats(perf_measures[1], projects, files_name[1])
        header = "Dataset \t|\tF3T\t|   K+S+RF\t|   K+S+LR\t|   K+S+SVM"
    print(rq + ": Statistical Results\n")
    for m in perf_measures:
        print(m)
        print(header)
        collect = pickle.load(open("stats_%s_%s.p" % (rq, m), "rb"))
        for p in projects:
            if len(p.upper()) < 9:
                str_proj = p.upper() + "\t"
            else:
                str_proj = p.upper()
            #print(collect[p].keys())
            for t in collect[p].keys():
                if t != "total":
                    str_proj += "\t,    %s(%s/%s)" % (int(collect[p][t]*100/collect[p]['total']), collect[p][t], collect[p]['total'])
            print(str_proj)
        print()



