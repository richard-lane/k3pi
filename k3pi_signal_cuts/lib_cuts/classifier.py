import numpy as np


class Classifier():
    """
    Basically just storange for an sklearn classifier
    returns predict_proba normalised to a mean of 0.5

    """

    def __init__(self, n_sig: int, n_bkg: int, clf):
        """
        Initialise the base classifier with *args and **kwargs and store the signal/bkg ratio
        so that we can use it later for predicting probabilities

        """
        self._sig_bkg_ratio = n_sig / n_bkg
        self.clf = clf

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        """
        Probability estimates as returned by base class predict_proba, but scaled such that
        the mean probability is 0 (i.e. removing the effect of an imbalanced training set)

        """
        return self.clf.predict_proba(points) / self._sig_bkg_ratio
