"""
A Model to contain classifiers, regressors, etc
"""
import os
import sys
import pandas as pd
from pdb import set_trace
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from smote import *

from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, Normalizer

root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

from metrics.abcd import ABCD


class PredictionModel:
    def __init__(self, classifier = "SVC"):
        self._set_classifier(classifier)

    def _set_classifier(self, classifier):
        if classifier == "SVC":
            self.clf = LinearSVC(C=1, dual=False)
        elif classifier == "RF":
            self.clf = RandomForestClassifier()

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
        dframe.loc[dframe[dframe.columns[-1]] > 0, dframe.columns[-1]] = 1
        dframe.loc[dframe[dframe.columns[-1]] == 0, dframe.columns[-1]] = 0
        return dframe

    def predict_defects(self, train, test, oversample=True, binarize=True):
        """
        Predict for Defects

        Parameters
        ----------
        train: numpy.ndarray or pandas.core.frame.DataFrame
            Training dataset as a pandas dataframe
        test: pandas.core.frame.DataFrame
            Test dataset as a pandas dataframe
        oversample: Bool
            Oversample with SMOTE
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

        x_train = train[train.columns[2:-1]].values
        y_train = train[train.columns[-1]].values
        #scaler = QuantileTransformer()
        #x_train = scaler.fit_transform(x_train)

        #if oversample:
        #   k = min(3, sum(y_train)-1)
        #   x_train, y_train = execute([k, 3], x_train, y_train)

        if oversample:
            k = min(3, sum(y_train)-1)
            print(k, sum(y_train)-1, x_train.shape[0], len(y_train))
            sm = SMOTE(kind='regular', k_neighbors=int(k))
        #    try:
        #        x_train, y_train = sm.fit_sample(x_train, y_train)
        #    except ValueError:
        #        k = min(3, sum(y_train) - 1)
        #        x_train, y_train = execute([k, 3], x_train, y_train)



        self.clf.fit(x_train, y_train)
        actual = test[test.columns[-1]].values
        x_test = test[test.columns[2:-1]]
        try:
            predicted = self.clf.predict(x_test)
        except:
            set_trace()
            predicted = [0]*actual
            print("FAIL")
        return actual, predicted
