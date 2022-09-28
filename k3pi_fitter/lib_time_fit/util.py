"""
Utilities that we might use in multiple places

"""
import sys
import pathlib
from collections import namedtuple
from typing import Tuple
import numpy as np
from . import definitions

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_efficiency.metrics import _counts

# No constraints
MixingParams = namedtuple("MixingParams", ("a", "b", "c"))

# Gaussian constraints on x and y
ConstraintParams = namedtuple("ConstraintParams", ("r_d", "b", "x", "y"))

# All params, for doing the scan
ScanParams = namedtuple("ScanParams", ("r_d", "x", "y", "re_z", "im_z"))


def bin_times(
    times: np.ndarray,
    weights: np.ndarray = None,
    bins: np.ndarray = definitions.TIME_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin times, possibly with weights - return counts and errors

    """
    if weights is None:
        weights = np.ones_like(times)

    # Turns out I already implemented this
    return _counts(times, weights, bins)


def ratio_err(
    numerator: np.ndarray,
    denominator: np.ndarray,
    num_err: np.ndarray,
    denom_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error of two arrays

    """
    ratio = numerator / denominator
    err = ratio * np.sqrt((num_err / numerator) ** 2 + (denom_err / denominator) ** 2)

    return ratio, err
