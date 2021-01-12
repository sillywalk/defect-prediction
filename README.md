# Defect Prediction
## Scientific Computation Software
[![Build Status](https://travis-ci.org/ai-se/se4sci.svg?branch=master)](https://travis-ci.org/ai-se/se4sci)
[![Coverage Status](https://coveralls.io/repos/github/se4sci/defect-prediction/badge.svg?branch=master)](https://coveralls.io/github/se4sci/defect-prediction?branch=master)![Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
![AUR](https://img.shields.io/aur/license/yaourt.svg)

This repository is the open source implementation of the EMBLEM paper work. 

### Requirements:
- my machine have both python2 and python3 installed. Python2 is for results analysis and python3 is for running the experiments. 
- most of the required python libs can be obtained by running ```pip install -r requirements.txt```.

### Process:
- Bug-fixing Commits Labeling: get github commits for each project and label them bug-fixing or no bug-fixing through this [folder](https://github.com/sillywalk/defect-prediction/tree/dev/data_labeling)
- Defect-introducing Commits Labels Retrieval: based on the labels from the previous step through SZZ from this [folder](https://github.com/sillywalk/defect-prediction/tree/dev/SZZ)
- Defect prediction: [comparing](https://github.com/sillywalk/defect-prediction/tree/dev/src) between all data miners across keyword generated labels and incremental AI+Human labels. 


### Statistical Results:
You can run the commands below for the numerical + statistical results for RQ2, 3, and 4 accordingly: 

```
bash rq2.sh 
bash rq3.sh
bash rq4.sh
```
