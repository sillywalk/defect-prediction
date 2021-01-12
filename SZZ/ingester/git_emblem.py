import os
import sys
import subprocess
import json
import logging
sys.path.append("..")
import math  # Required for the math.log function
from commitFile import CommitFile  # Represents a file
#from . import classifier
from classifier.classifier import *  # Used for classifying each commit
import time
import pdb
import pandas as pd
​
"""
file: repository.py
authors: Ben Grawi <bjg1568@rit.edu>, Christoffer Rosen <cbr4830@rit.edu>
date: October 2013
description: Holds the repository git abstraction class
"""
​
​
class Git():
    """
    Git():
    pre-conditions: git is in the current PATH
                    self.path is set in a parent class
    description: a very basic abstraction for using git in python.
    """
    # Two backslashes to allow one backslash to be passed in the command.
    # This is given as a command line option to git for formatting output.
​
    # A commit mesasge in git is done such that first line is treated as the subject,
    # and the rest is treated as the message. We combine them under field commit_message
​
    # We want the log in ascending order, so we call --reverse
    # Numstat is used to get statistics for each commit
    LOG_FORMAT = '--pretty=format:\" CAS_READER_STARTPRETTY\
    \\"parent_hashes\\"CAS_READER_PROP_DELIMITER: \\"%P\\",CAS_READER_PROP_DELIMITER2\
    \\"commit_hash\\"CAS_READER_PROP_DELIMITER: \\"%H\\",CAS_READER_PROP_DELIMITER2\
    \\"author_name\\"CAS_READER_PROP_DELIMITER: \\"%an\\",CAS_READER_PROP_DELIMITER2\
    \\"author_email\\"CAS_READER_PROP_DELIMITER: \\"%ae\\",CAS_READER_PROP_DELIMITER2\
    \\"author_date\\"CAS_READER_PROP_DELIMITER: \\"%ad\\",CAS_READER_PROP_DELIMITER2\
    \\"author_date_unix_timestamp\\"CAS_READER_PROP_DELIMITER: \\"%at\\",CAS_READER_PROP_DELIMITER2\
    \\"commit_message\\"CAS_READER_PROP_DELIMITER: \\"%s%b\\"\
    CAS_READER_STOPPRETTY \" --numstat --reverse '
​
    SHOW_ONE_COMM = 'git show {v} --pretty=format:\" \
    \\"parent_hashes\\"CAS_READER_PROP_DELIMITER: \\"%P\\",CAS_READER_PROP_DELIMITER2\
    \\"commit_hash\\"CAS_READER_PROP_DELIMITER: \\"%H\\",CAS_READER_PROP_DELIMITER2\
    \\"author_name\\"CAS_READER_PROP_DELIMITER: \\"%an\\",CAS_READER_PROP_DELIMITER2\
    \\"author_email\\"CAS_READER_PROP_DELIMITER: \\"%ae\\",CAS_READER_PROP_DELIMITER2\
    \\"author_date\\"CAS_READER_PROP_DELIMITER: \\"%ad\\",CAS_READER_PROP_DELIMITER2\
    \\"author_date_unix_timestamp\\"CAS_READER_PROP_DELIMITER: \\"%at\\",CAS_READER_PROP_DELIMITER2\
    \\"commit_message\\"CAS_READER_PROP_DELIMITER: \\"%s%b\\" \
    CAS_READER_STOPPRETTY \" --numstat '
    DIFF_CMD = 'git diff {s}..{s} --numstat'
    CLONE_CMD = 'git clone {!s} {!s}'  # git clone command w/o downloading src code
    PULL_CMD = 'git pull'  # git pull command
    RESET_CMD = 'git reset --hard FETCH_HEAD'
    CLEAN_CMD = 'git clean -df'  # f for force clean, d for untracked directories
​
    REPO_DIRECTORY = "/CASRepos/git/"  # directory in which to store repositories
    count = 0
​
​
    def getDiffStatsProperties(self, stats, commitFiles, devExperience, author, unixTimeStamp):
        """
                getCommitStatsProperties
                Helper method for log. Caclulates statistics for each change/commit and
                returns them as a comma seperated string. Log will add these to the commit object
                properties
​
                @param stats            These are the stats given by --numstat as an array
                @param commitFiles      These are all tracked commit files
                @param devExperience    These are all tracked developer experiences
                @param author           The author of the commit
                @param unixTimeStamp    Time of the commit
                """
​
        statProperties = ""
​
        # Data structures to keep track of info needed for stats
        subsystemsSeen = []  # List of system names seen
        directoriesSeen = []  # List of directory names seen
        locModifiedPerFile = []  # List of modified loc in each file seen
        authors = []  # List of all unique authors seen for each file
        fileAges = []  # List of the ages for each file in a commit
​
        # commit.guru metrics per file
        files_metrics = []
        temp_total_LOC = 0
        temp_locModifiedPerFile = []
        temp_nf = 0
​
        # Stats variables
        la = 0  # lines added
        ld = 0  # lines deleted
        nf = 0  # Number of modified files
        ns = 0  # Number of modified subsystems
        nd = 0  # number of modified directories
        entrophy = 0  # entrophy: distriubtion of modified code across each file
        lt = 0  # lines of code in each file (sum) before the commit
        ndev = 0  # the number of developers that modifed the files in a commit
        age = 0  # the average time interval between the last and current change
        exp = 0  # number of changes made by author previously
        rexp = 0  # experience weighted by age of files ( 1 / (n + 1))
        sexp = 0  # changes made previous by author in same subsystem
        totalLOCModified = 0  # Total modified LOC across all files
        nuc = 0  # number of unique changes to the files
        filesSeen = ""  # files seen in change/commit
​
        for stat in stats:
            file_m = {"File": "", "la": 0, "ld": 0, "ns": 0, "ns": 0, "nd": 0, "nuc": 0,
                      "entrophy": 0, "lt": 0, "ndev": 0, "age": 0,
                      "exp": 0, "rexp": 0, "sexp": 0, "fix": 0}

            if (stat == ' ' or stat == ''):
                continue

            fileStat = stat.split("\\t")

            # Check that we are only looking at file stat (i.e., remove extra newlines)
            if (len(fileStat) < 2):
                continue

            # catch the git "-" line changes
            try:
                fileLa = int(fileStat[0])
                fileLd = int(fileStat[1])
            except:
                fileLa = 0
                fileLd = 0

            # Remove oddities in filename so we can process it
            fileName = (fileStat[2].replace("'", '').replace('"', '').replace("\\", ""))
            file_m['File'] = fileName
            file_m['la'] = fileLa
            file_m['ld'] = fileLd
            totalModified = fileLa + fileLd

            # have we seen this file already?
            if (fileName in commitFiles):
                prevFileChanged = commitFiles[fileName]
                prevLOC = getattr(prevFileChanged, 'loc')
                prevAuthors = getattr(prevFileChanged, 'authors')
                prevChanged = getattr(prevFileChanged, 'lastchanged')
                file_nuc = getattr(prevFileChanged, 'nuc')
                nuc += file_nuc
                lt += prevLOC
                file_m['nuc'] = file_nuc
                file_m['lt'] = prevLOC
​
                for prevAuthor in prevAuthors:
                    if prevAusexpthor not in authors:
                        authors.append(prevAuthor)
​
                # Convert age to days instead of seconds
                age += ((int(unixTimeStamp) - int(prevChanged)) / 86400)
                fileAges.append(prevChanged)
​
                # Update the file info
​
                file_nuc += 1  # file was modified in this commit
                setattr(prevFileChanged, 'loc', prevLOC + fileLa - fileLd)
                setattr(prevFileChanged, 'authors', authors)
                setattr(prevFileChanged, 'lastchanged', unixTimeStamp)
                setattr(prevFileChanged, 'nuc', file_nuc)
​
            else:
​
                # new file we haven't seen b4, add it to file commit files dict
                if (author not in authors):
                    authors.append(author)
​
                if (unixTimeStamp not in fileAges):
                    fileAges.append(unixTimeStamp)
​
                fileObject = CommitFile(fileName, fileLa - fileLd, authors, unixTimeStamp)
                commitFiles[fileName] = fileObject
​
            # end of stats loop
​
            locModifiedPerFile.append(totalModified)  # Required for entrophy
            totalLOCModified += totalModified
            fileDirs = fileName.split("/")
​
            if (len(fileDirs) == 1):
                subsystem = "root"
                directory = "root"
            else:
                subsystem = fileDirs[0]
                directory = "/".join(fileDirs[0:-1])
​
            if (subsystem not in subsystemsSeen):
                subsystemsSeen.append(subsystem)
​
            if (author in devExperience):
                experiences = devExperience[author]
                exp += sum(experiences.values())
​
                if (subsystem in experiences):
                    sexp = experiences[subsystem]
                    experiences[subsystem] += 1
                else:
                    experiences[subsystem] = 1
​
                try:
                    rexp += (1 / (age) + 1)
                except:
                    rexp += 0
​
            else:
                devExperience[author] = {subsystem: 1}
​
            if (directory not in directoriesSeen):
                directoriesSeen.append(directory)
​
            # Update file-level metrics
            la += fileLa
            ld += fileLd
            nf += 1
            filesSeen += fileName + ",CAS_DELIMITER,"
            if file_m['File'] and "__init__.py" not in file_m['File']:
                #pdb.set_trace()
                type_f = file_m['File'].split(".")
                if type_f:
                    if type_f[-1].upper() in set(["C", "CC", "CPP", "CXX", "H", "HPP",
                                          "F90", "F", "F77", "F03", "F95", "FOR", "FTN",
                                          "PY", "JAVA"]):
                        temp_nf += 1
                        files_metrics.append(file_m)
                        temp_total_LOC += totalModified
                        temp_locModifiedPerFile.append(totalModified)
​
        # End stats loop
​
        if (nf < 1):
            return "", None
​
        # Update commit-level metrics
        ns = len(subsystemsSeen)
        nd = len(directoriesSeen)
        ndev = len(authors)
​
        temp_entrophy = 0
        for i in range(len(files_metrics)):
            files_metrics[i]['ns'] = ns
            files_metrics[i]['nd'] = nd
            files_metrics[i]['ndev'] = ndev
            files_metrics[i]['age'] = float(age) / temp_nf
            files_metrics[i]['exp'] = float(exp) / temp_nf
            files_metrics[i]['rexp'] = float(rexp) / temp_nf
            temp_fileLocMod = temp_locModifiedPerFile[i]
            if temp_fileLocMod != 0:
                avg = float(temp_fileLocMod) / temp_total_LOC
                temp_entrophy -= (avg * math.log(avg, 2))
                files_metrics[i]['entrophy'] = 0 - (avg * math.log(avg, 2))
​
        if files_metrics:
            self.count += 1
​
        lt = lt / nf
        age = age / nf
        exp = exp / nf
        rexp = rexp / nf
        # Update entrophy
        for fileLocMod in locModifiedPerFile:
            if fileLocMod != 0:
                avg = fileLocMod / totalLOCModified
                entrophy -= (avg * math.log(avg, 2))
        # print("testing entrophy", temp_entrophy, entrophy)
        # Add stat properties to the commit object
        statProperties += ',"la":"' + str(la) + '\"'
        statProperties += ',"ld":"' + str(ld) + '\"'
        statProperties += ',"fileschanged":"' + filesSeen[0:-1] + '\"'
        statProperties += ',"nf":"' + str(nf) + '\...
