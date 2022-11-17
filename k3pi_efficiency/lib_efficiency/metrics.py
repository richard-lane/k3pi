"""
Measurements and things that we care about

"""
import logging
from typing import Tuple
import numpy as np


def _counts(t: np.ndarray, wt: np.ndarray, bins: np.ndarray) -> Tuple:
    """
    Returns the counts in each bin and their errors

    :param t: times
    :param wt: weights
    :param bins: bins

    :return: counts in each bin
    :return: errors on counts

    """
    logging.warning(
        "Use the counts fcn in lib_data.stats instead", exc_info=DeprecationWarning()
    )

    indices = np.digitize(t, bins) - 1

    # Underflow
    if -1 in indices:
        raise ValueError(f"Underflow: bins from {bins[0]}; {np.min(t)=}")

    # Overflow
    if len(bins) - 1 in indices:
        raise ValueError("Overflow")

    n_bins = len(bins) - 1

    # Init with NaN so its obvious if something has gone wrong
    counts = np.ones(n_bins) * np.nan
    errs = np.ones(n_bins) * np.nan

    for i in range(n_bins):
        mask = indices == i

        counts[i] = np.sum(wt[mask])
        errs[i] = np.sqrt(np.sum(wt[mask] ** 2))

    return counts, errs
