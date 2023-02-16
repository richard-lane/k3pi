"""
Definitions of some things

"""
from typing import Tuple
import numpy as np
from . import pdfs


def mass_bins(n_bins: int = 100) -> np.ndarray:
    """
    Bins used for mass fit

    :param n_bins: number of bins
    :returns: the bins

    """
    return np.linspace(*pdfs.domain(), n_bins + 1)


def reduced_mass_bins(n_bins: int = 100) -> np.ndarray:
    """
    Bins used for mass fit over reduced domain

    :param n_bins: number of bins
    :returns: the bins

    """
    return np.linspace(*pdfs.reduced_domain(), n_bins + 1)


def nonuniform_mass_bins(boundaries: Tuple, num_bins: Tuple) -> np.ndarray:
    """
    Nonuniform bins for the mass fitter

    :param boundaries: tuple of floats denoting where the boundaries
                       between regions is
    :param num_bins: number of bins in each region

    :returns: bins in the range pdfs.domain()

    """

    assert len(boundaries) == len(num_bins) - 1

    low, high = pdfs.domain()
    boundaries = tuple((low, *boundaries, high))

    return np.unique(
        np.concatenate(
            [
                np.linspace(a, b, n)
                for (a, b, n) in zip(boundaries[:-1], boundaries[1:], num_bins)
            ]
        )
    )
