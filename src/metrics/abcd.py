from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from pdb import set_trace
import pandas as pd
from sklearn.metrics import auc

class ABCD:
    def __init__(self, actual, predicted, loc):
        self.loc = pd.DataFrame(loc)
        self.actual = pd.DataFrame(actual, columns=['Actual'])
        self.predicted = pd.DataFrame(predicted, columns=['Predicted'])
        self.dframe = pd.concat([self.actual, self.predicted, self.loc], axis=1)
        self.dframe['InspectedLOC'] = self.dframe.CountLine.cumsum()
        self._set_aux_params()
    
    def _set_aux_params(self):
        self.M = len(self.dframe)
        self.N = self.dframe.Actual.sum()
        inspected_max = self.dframe.InspectedLOC.max()
        for i in range(self.M):
            if self.dframe.InspectedLOC.iloc[i] >= 0.2 * inspected_max:
                # If we have inspected more than 20% of the total LOC
                # break
                break

        self.inspected_20 = self.dframe.iloc[:i]
        # Number of changes when we inspect 20% of LOC
        self.m = len(self.inspected_20)
        self.n = self.inspected_20.Predicted.sum()

    def get_pd_pf(self, ):
        """
        Obtain Recall (Pd) and False Alarm (Pf) scores

        Parameters
        ----------
        actual: numpy.ndarray, shape=[n_samples]
            Ground truth (correct) target values.
        predicted: numpy.ndarray, shape=[n_samples]
            Estimated targets as returned by a classifier.
        
        Returns
        -------
        pd: float
            Recall (pd) value 
        pf: float
            False alarm (pf) values 
        """
        tn, fp, fn, tp = confusion_matrix(
            self.inspected_20.Actual, self.inspected_20.Predicted).ravel()
        pd = int(100 * tp/(tp+fn+1e-5))
        pf = int(100 * fp/(fp+tn+1e-5))

        return pd, pf

    def get_f_score(self, beta=1):
        """
        Obtain F scores

        Parameters
        ----------
        actual: numpy.ndarray, shape=[n_samples]
            Ground truth (correct) target values.
        predicted: numpy.ndarray, shape=[n_samples]
            Estimated targets as returned by a classifier.
        beta: float, default=1
            Amount by which recall is weighted higher than precision
        
        Returns
        -------
        prec: float
            Precision
        f: float
            F score 
        """

        tn, fp, fn, tp = confusion_matrix(
            self.inspected_20.Actual, self.inspected_20.Predicted).ravel()
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f = int(100 * (1 + beta**2) * (prec * recall) /
                (beta**2 * prec + recall + 1e-5))
        prec = int(100 * prec)
        recall = int(100 * recall)
        return prec, f
    
    def get_pci_20(self):
        """
        Proportion of Changes Inspected when 20% LOC modified by all changes are 
        inspected. A high PCI@k% indicates that, under the same number of LOC to 
        inspect, developers need to inspect more changes.
        
        Returns
        -------
        int:
            The PCI value
        """        
        pci_20 = int(self.m / self.M * 100)
        return pci_20
    
    def get_ifa(self):
        """
        Inital False Alarm
        
        Number of Initial False Alarms encountered before we find the first 
        defect. 
        
        Parameters
        ----------
        actual : array_like
            Actual labels
        predicted : array_like
            Predicted labels
        
        Returns
        -------
        int:
            The IFA value
        
        Notes
        -----
        We compute the IFA by sorting the actual bug counts, and then computing
        the number of false alarms until the first true positive is discovered.

        The value is normalized to a percentage value.
        """

        for i in range(len(self.dframe)):
            if self.dframe['Actual'].iloc[i] == self.dframe['Predicted'].iloc[i] == 1:
                break

        pred_vals = self.dframe['Predicted'].values[:i]
        ifa = int(sum(pred_vals)/(i+1) * 100)
        return i
