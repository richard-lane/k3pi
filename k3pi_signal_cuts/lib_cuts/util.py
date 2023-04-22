"""
Utility functions

"""
from typing import Iterable

import numpy as np


def weight(label: np.ndarray, signal_fraction: float, evt_wts: np.ndarray) -> float:
    """
    The weight we need to apply to each signal event to get the desired signal fraction

    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :return: the weight to apply to each signal event to get the desired signal fraction

    """
    n_sig, n_bkg = np.sum(evt_wts[label == 1]), np.sum(evt_wts[label == 0])

    retval = n_bkg * signal_fraction / (n_sig * (1 - signal_fraction))

    # If either of these asserts fail, you might be using
    # the wrong size inputs.
    # See tests/test_signal_cuts.py:test_sig_frac_overflow
    # for a potential explanation
    # ^Not entirely sure what the above comment means but it might mean
    # the reweighting would need to scale UP the amount of MC, instead of
    # scale it down
    # If this is the case we should really scale down the amount of uppermass instead
    assert 0.0 < retval
    assert retval < 1.0

    return retval


def weights(
    label: np.ndarray, signal_fraction: float, evt_wts: np.ndarray = None
) -> np.ndarray:
    """
    Weights to apply to the sample so that we get the right proportion of
    signal/background

    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :param evt_wts: weights to apply to each event - alters the statistics

    :return: array of weights; 1 for bkg events, the right number for sig
             events such that there's the correct proportion of sig + bkg

    """
    if evt_wts is None:
        evt_wts = np.ones_like(label)

    retval = np.ones(len(label), dtype=float)

    # Since signal is labelled with 1
    retval[label == 1] = weight(label, signal_fraction, evt_wts)

    return retval


def resample_mask(
    gen: np.random.Generator,
    label: np.ndarray,
    signal_fraction: float,
    evt_wts: np.ndarray = None,
) -> np.ndarray:
    """
    Boolean mask of which labels to keep to achieve the right signal fraction

    :param rng: random number generator
    :param label: class labels - 0 for bkg, 1 for signal
    :param signal_fraction: desired proportion (signal / signal + bkg)
    :return: mask of which labels to keep which will give us the right
             signal fraction

    """
    if evt_wts is None:
        evt_wts = np.ones_like(label)

    # Keep all the bkg evts (label == 0); throw away some of the signal evts (label == 1)
    retval = np.ones(len(label), dtype=np.bool_)

    discard = (label == 1) & (
        gen.random(len(label)) > weight(label, signal_fraction, evt_wts)
    )
    retval[discard] = False

    return retval


def discard(gen: np.random.Generator, iterable: Iterable, n_desired: float) -> Iterable:
    """
    Randomly throw away entries from an iterable to have approximately n_desired entries

    """
    n_tot = len(iterable)

    return iterable[gen.random(n_tot) < (n_desired / n_tot)]
