"""
Metrics, numbers etc. that we can use to check the classifier is behaving correctly

"""
import numpy as np


def signal_significance(n_sig: np.ndarray, n_bkg: np.ndarray) -> np.ndarray:
    """
    Signal significance

    Ideally we would keep as much signal and lose as much background as possible - this is usually
    (in HEP) achieved by optimising s/sqrt(s + b) for signal s and background b

    In this case, we want to optimise the signal significance of our testing sample (and of the real
    data, but there's no way to measure that)

    :param n_sig: array of numbers of signal events
    :param n_bkg: array of numbers of background events
    :returns: array of signal significances

    """
    return n_sig / np.sqrt(n_sig + n_bkg)
