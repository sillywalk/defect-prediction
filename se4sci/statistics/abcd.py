from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

class ABCD:
    def __init__(self):
        pass
    
    @classmethod
    def get_pd_pf(self, actual, predicted):
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
        (pd, pf): tuple(float, float)
            A tuple of recall (pd) and false alarm (pf) values 
        """
        