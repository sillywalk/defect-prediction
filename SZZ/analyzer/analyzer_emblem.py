"""
file: Analyzer.py
author: Christoffer Rosen <cbr4830@rit.edu>
date: November 2013
description: This module contains the functions for analyzing a repo with a given id.
Currently only supports the GitHub Issue Tracker.
"""
import sys
sys.path.append("..")
from datetime import datetime, timedelta
from orm.repository import *
from orm.commit import *
from bugfinder import *
from metricsgenerator import *
from githubissuetracker import *
from caslogging import logging
from notifier import *
from config import config
import pdb
from git_commit_linker import *
from sqlalchemy import Date, cast
​
def analyze(repo_id):
    """
    Analyze the repository with the given id. Gets the repository from the repository table
    and starts ingesting using the analyzeRepo method.
    @param repo_id		The repository id to analyze
    """
    session = Session()
​
    repo_to_analyze = (session.query(Repository)
                .filter (Repository.id == repo_id)
                .all()
                )
​
    # Verify that repo exists
    if len(repo_to_analyze) > 0:
        analyzeRepo(repo_to_analyze[0],session)
    else:
        logging.info('Repo with id ' + repo_id_to_analyze + ' not found!')
​
    session.close()
​
def analyzeRepo(repository_to_analyze, proj_commits, type_bug):
    """
    Analyzes the given repository
    @param repository_to_analyze	The repository to analyze.
    @param session                  SQLAlchemy session
    @private
    """
    #repo_name = repository_to_analyze.name
    repo_id = repository_to_analyze.id
    #last_analysis_date = repository_to_analyze.analysis_date
​
    # Update status of repo to show it is analyzing
    #repository_to_analyze.status = "Analyzing"
    #session.commit()
​
    #print('Worker analyzing repository id ' + repo_id)
​
    # all commits in descending order
    # all_commits = sorted(proj_commits, key=lambda x: int(x['author_date_unix_timestamp']), reverse=True)
    # proj_commits['time'] = pd.to_datetime(proj_commits['time'])
    corrective_commits = proj_commits[proj_commits[type_bug] == 1]
    all_commits = proj_commits.sort_values(by=['time'], ascending=False)
    all_commits = make_list_dict(all_commits, type_bug)
    corrective_commits = make_list_dict(corrective_commits, type_bug)
    #corrective_commits = [c for c in corrective_commits if c['fix'] == "True"]
    git_commit_linker = GitCommitLinker(repo_id)
    corrective_commits_linked = []
    print("Linking " + str(len(corrective_commits)) + " new corrective commits for repo " + repo_id)
​
    try:
        corrective_commits_linked = git_commit_linker.linkCorrectiveCommits(corrective_commits, all_commits)
        #pdb.set_trace()
    except Exception as e:
        print("Got an exception linking bug fixing changes to bug inducing changes for repo " + repo_id)
​
    fixes = []
    for index in range(len(corrective_commits_linked) - 1, -1, -1):
        comm_link = corrective_commits_linked[index]
        comm_fix = ""
        #pdb.set_trace()
        if "fixes" in comm_link.keys():
            for c in comm_link['fixes']:
                comm_fix += c + "<@@@>"
        fixes.append(comm_fix)
​
    proj_commits[type_bug + '_fixes'] = fixes
    proj_commits.to_csv("../ingester/CASRepos/git/%s/%s_commits.csv" % (repo.id, repo.id), index=False)
    with open("../ingester/CASRepos/git/%s/%s_%s_commits_linked.json" % (repo.id, repo.id, type_bug), 'w') as outfile:
        json.dump(corrective_commits_linked, outfile)
​
​
    '''
    #repository_to_analyze.status = "Error"
    #session.commit() # update repo status
    #raise
    
    # Signify to CAS Manager that this repo is ready to have it's model built
    if repository_to_analyze.status != "Error":
        repository_to_analyze.status = "In Queue to Build Model"
        session.commit() # update repo status
    '''
​
​
def make_list_dict(p_df, type_bug):
    p_df_dict = []
    for index, row in p_df.iterrows():
        dict_entry = {"commit_hash": row['hash'].strip(), "time": str(row['time'])}
        if row[type_bug] == 1:
            dict_entry["fix"] = True
        else:
            dict_entry["fix"] = False
        p_df_dict.append(dict_entry)
    return p_df_dict
​
​
if __name__ == "__main__":
    import pandas as pd
    repo_url = "https://github.com/" + sys.argv[1]
    repo_id = sys.argv[2]
    from datetime import datetime
    repo = type('obj', (object,), {'id': repo_id,
                                   'url': repo_url,
                                   'ingestion_date': None})
    all_commits = pd.read_csv("../ingester/CASRepos/git/%s/%s_commits.csv" % (repo_id, repo_id))
    all_commits['hash'] = all_commits['hash'].astype(str)
    for type_bug in ['keyword_buggy', 'fastread_buggy']:
        analyzeRepo(repo, all_commits, type_bug)
        print("*" * 50)
        print(repo_id, type_bug)
        print("*" * 50)
