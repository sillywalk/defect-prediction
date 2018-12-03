from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from pdb import set_trace
import pandas as pd
from numpy import trapz
import numpy as np
from sklearn.metrics import auc

class ABCD:
    def __init__(self, actual, predicted, loc):
        self.loc = pd.DataFrame(loc)
        self.actual = pd.DataFrame(actual, columns=['Actual'])
        self.predicted = pd.DataFrame(predicted, columns=['Predicted'])
        self.dframe = pd.concat([self.actual, self.predicted, self.loc], axis=1)
        self.dframe['InspectedLOC'] = self.loc.cumsum().iloc[:,0]
        self.inspected_perc = {'datasets': [], 'Pf': [], 'Pd': [], 'Prec': [], 'F1': [], 'G1': []}
        self._set_aux_params(.2)
        try:
            self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.inspected_20.Actual, self.inspected_20.Predicted).ravel()
        except ValueError:
            self._set_aux_params(.5)
            self.tn, self.fp, self.fn, self.tp = self.manual_count(self.inspected_20.Actual.values.tolist(),
                                                                   self.inspected_20.Predicted.values.tolist())

    def get_G(self):
        pd = 1 * self.tp / (self.tp + self.fn + 1e-5)
        pf = 1 * self.fp / (self.fp + self.tn + 1e-5)
        g = int((2 * 100 * pd * (1 - pf))/(1 + pd - pf + 1e-5))
        return g

    def get_score_total(self, beta=1):
        """
        Obtain metric scores for investigating everything

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
            self.actual.values, self.predicted.values).ravel()
        prec = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f = int(100 * (1 + beta ** 2) * (prec * recall) /
                (beta ** 2 * prec + recall + 1e-5))
        prec = int(100 * prec)
        recall = int(100 * recall)
        #print("hello", prec, f)
        return prec, f

    def _set_aux_params(self, perc):
        self.M = len(self.dframe)
        self.N = self.dframe.Actual.sum()
        inspected_max = self.dframe.InspectedLOC.max()
        for i in range(self.M):
            if self.dframe.InspectedLOC.iloc[i] >= (perc * inspected_max):
                # If we have inspected more than 20% of the total LOC
                # break
                break

        self.inspected_20 = self.dframe.iloc[:i]
        # Number of changes when we inspect 20% of LOC
        self.m = len(self.inspected_20)
        self.n = self.inspected_20.Predicted.sum()

    def manual_count(self, actual, predicted, indx=1):
        TP, TN, FP, FN = 0, 0, 0, 0
        for a, b in zip(actual, predicted):
            if a == indx and b == indx:
                TP += 1
            elif a == b and a != indx:
                TN += 1
            elif a != indx and b == indx:
                FP += 1
            elif a == indx and b != indx:
                FN += 1
        return TN, FP, FN, TP

    def _set_auc(self, h):
        self.M = len(self.dframe)
        self.N = self.dframe.Actual.sum()
        inspected_max = self.dframe.InspectedLOC.max()

        threshold_loc = 0
        for j in range(h):
            threshold_loc += 0.5 / h
            for i in range(self.M):
                if self.dframe.InspectedLOC.iloc[i] >= (threshold_loc * inspected_max):
                    self.inspected_perc['datasets'].append(self.dframe.iloc[:i])
                    try:
                        tn, fp, fn, tp = confusion_matrix(self.inspected_perc['datasets'][-1].Actual, self.inspected_perc['datasets'][-1].Predicted).ravel()
                    except ValueError:
                        tn, fp, fn, tp = self.manual_count(self.inspected_perc['datasets'][-1].Actual.values.tolist(),
                                                    self.inspected_perc['datasets'][-1].Predicted.values.tolist())
                    prec = tp / (tp + fp + 1e-5)
                    pd = tp / (tp + fn + 1e-5)
                    pf = 1 * fp / (fp + tn + 1e-5)
                    self.inspected_perc['F1'].append(int(100 * (1 + 1 ** 2) * (prec * pd) /
                            (1 ** 2 * prec + pd + 1e-5)))
                    self.inspected_perc['G1'].append(int((2 * 100 * pd * (1 - pf)) / (1 + pd - pf + 1e-5)))
                    self.inspected_perc['Pf'].append(int(100 * pf))
                    self.inspected_perc['Prec'].append(int(100 * prec))
                    self.inspected_perc['Pd'].append(int(100 * pd))
                    # If we have inspected more than 20% of the total LOC
                    break
        return trapz(np.array(self.inspected_perc['Pd']), dx=0.5 / h), \
               trapz(np.array(self.inspected_perc['Pf']), dx=0.5 / h), \
               trapz(np.array(self.inspected_perc['Prec']), dx=0.5 / h), \
               trapz(np.array(self.inspected_perc['F1']), dx=0.5 / h), \
               trapz(np.array(self.inspected_perc['G1']), dx=0.5 / h)




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
        pd = int(100 * self.tp/(self.tp+self.fn+1e-5))
        pf = int(100 * self.fp/(self.fp+self.tn+1e-5))

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
        prec = self.tp / (self.tp + self.fp + 1e-5)
        recall = self.tp / (self.tp + self.fn + 1e-5)
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
