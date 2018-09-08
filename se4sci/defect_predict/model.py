"""
A Model to contain classifiers, regressors, etc
"""
import os
import sys
import pandas as pd
from pdb import set_trace
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path

root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from statistics.metrics import ABCD

class PredictionModel:
    def __init__(self): pass
    
    @staticmethod
    def _binarize(dframe):
        """
        Turn the dependent variable column to a binary class

        Parameters
        ----------
        dframe: pandas.core.frame.DataFrame
            A pandas dataframe with independent and dependent variable columns
        
        Return
        ------
        dframe: pandas.core.frame.DataFrame
            The orignal dataframe with binary dependent variable columns
        """
        dframe.loc[dframe[dframe.columns[-1]] > 0, dframe.columns[-1]] = True
        dframe.loc[dframe[dframe.columns[-1]] == 0, dframe.columns[-1]] = False
        return dframe

    def predict_defects(self, train, test, binarize=True):
        """
        Predict for Defects

        Parameters
        ----------
        train: numpy.ndarray or pandas.core.frame.DataFrame
            Training dataset as a pandas dataframe
        test: pandas.core.frame.DataFrame
            Test dataset as a pandas dataframe
        binarize: Bool
            A boolean variable to 

        Return
        ------
        actual: numpy.ndarray
            Actual defect counts
        predicted: numpy.ndarray
            Predictied defect counts 
        """

        if binarize:
            train = self._binarize(train)
            test = self._binarize(test)
        
        x_train = train[train.columns[1:-1]]
        y_train = train[train.columns[-1]]
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        actual = test[test.columns[-1]].values
        x_test = test[test.columns[1:-1]]
        predicted = clf.predict(x_test)
        pd, pf = ABCD.get_pd_pf(actual, predicted)
        return actual, predicted




