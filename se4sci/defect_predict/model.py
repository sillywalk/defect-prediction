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

class PredictionModel:
    def __init__(self): pass
    
    @staticmethod
    def _binarize(self, dframe):
        """
        Turn the dependent variable column to a binary class

        Parameters
        ----------
        dframe: pandas.core.frame.DataFrame
            A pandas dataframe with independent and dependent variable columns
        
        Return
        ------
        bin_dframe: pandas.core.frame.DataFrame
            The orignal dataframe with binary dependent variable columns
        """
        bin_dframe = dframe.loc[dframe[dframe.columns[-1]] > 0, dframe.columns[-1]] = True
        bin_dframe = dframe.loc[dframe[dframe.columns[-1]] == 0, dframe.columns[-1]] = False
        return bin_dframe

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
        actual: numpy.array
            Actual defect counts
        predicted: numpy.array
            Predictied defect counts 
        """
        if binarize:
            train = self._binarize(train)
            test = self._binarize(test)
        
        x_train = train[train.columns[:-1]]
        y_train = train[train.columns[-1]]
        clf = RandomForestClassifier()
        clf.fit(X, y)
        actual = test[test.columns[-1]]
        x_test = test[test.columns[:-1]]
        predicted = clf.predict(x_test)
        set_trace()



