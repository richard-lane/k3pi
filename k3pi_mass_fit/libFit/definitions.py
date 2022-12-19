"""
Definitions of some things

"""
import numpy as np
from . import pdfs


def mass_bins(n_bins: int = 100) -> np.ndarray:
    """
    Bins used for mass fit

    :param n_bins: number of bins
    :returns: the bins

    """
    return np.linspace(*pdfs.domain(), n_bins + 1)
