"""
Utility functions

"""
import numpy as np


def weight(label: np.ndarray, signal_fraction: float) -> float:
    """
    The weight we need to apply to each signal event to get the desired signal fraction

    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :return: the weight to apply to each signal event to get the desired signal fraction

    """
    n_sig, n_bkg = np.sum(label == 1), np.sum(label == 0)

    retval = n_bkg * signal_fraction / (n_sig * (1 - signal_fraction))
    assert 0.0 < retval and retval < 1.0

    return retval


def weights(label: np.ndarray, signal_fraction: float) -> np.ndarray:
    """
    Weights to apply to the sample so that we get the right proportion of
    signal/background

    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :return: array of weights; 1 for bkg events, the right number for sig
             events such that there's the correct proportion of sig + bkg

    """
    retval = np.ones(len(label))

    # Since signal is labelled with 1
    retval[label == 1] = weight(label, signal_fraction)

    return retval


def resample_mask(
    gen: np.random.Generator, label: np.ndarray, signal_fraction: float
) -> np.ndarray:
    """
    Boolean mask of which labels to keep to achieve the right signal fraction

    :param rng: random number generator
    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :return: mask of which labels to keep which will give us the right
             signal fraction

    """
    # Keep all the bkg evts (label == 0); throw away some of the signal evts (label == 1)
    retval = np.ones(len(label), dtype=np.bool_)

    discard = (label == 1) & (gen.random(len(label)) > weight(label, signal_fraction))
    retval[discard] = False

    return retval
