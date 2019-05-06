"""
A Defect Prediction model for file level metrics
"""
import os
import re
import sys
import pandas
import numpy as np
from pdb import set_trace
from prettytable import PrettyTable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from pathlib import Path
root = Path.cwd()

while root.name is not 'src':
    # Climb up the directory tree until you reach
    root = root.parent

if root not in sys.path:
    sys.path.append(root)

from metrics.abcd import ABCD
from data.data_handler import DataHandler
from prediction.model import PredictionModel

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dh = DataHandler()
    data = dh.get_data(top_k=1)
    for _, val in data.items():
        data = val
    
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X = model.transform(X)
    pca = PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    colors = ['navy', 'darkorange']

    for X_transformed, title in [(X, "PCA")]:
        plt.figure(figsize=(8, 8))
        for color, i, target_name in zip(colors, [0, 1], ['Buggy', 'Clean']):
            plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                        color=color, lw=2, label=target_name)
        
        plt.title(title + " JQuery")
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        
    plt.show()

    set_trace()

