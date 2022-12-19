"""
Functions for sWeighting

https://cds.cern.ch/record/2777845/files/QCHS2021_MKenzie.pdf
"""
from typing import Tuple, Callable, Iterable
import numpy as np
import pandas as pd
from scipy.linalg import solve

from . import pdfs, definitions


def sig(delta_m: np.ndarray, params: Tuple) -> np.ndarray:
    """signal pdf, given params"""
    return pdfs.normalised_signal(delta_m, *params[2:-2])


def bkg(delta_m: np.ndarray, params: Tuple) -> np.ndarray:
    """bkg pdf, given params"""
    return pdfs.normalised_bkg(delta_m, *params[-2:])


def overall(delta_m: np.ndarray, params: Tuple) -> np.ndarray:
    """overall pdf, given params"""
    return pdfs.model(delta_m, *params) / (params[0] + params[1])


def w_matrix(params: Tuple) -> np.ndarray:
    """
    Evaluate the W-matrix for the sWeighting

    """
    # Find the signal and background pdfs from our params

    # Construct matrix + return
    retval = np.zeros((2, 2))
    pts = definitions.mass_bins(1000)

    retval[0, 0] = np.trapz(sig(pts, params) ** 2 / overall(pts, params), x=pts)

    retval[1, 1] = np.trapz(bkg(pts, params) ** 2 / overall(pts, params), x=pts)

    retval[0, 1] = retval[1, 0] = np.trapz(
        sig(pts, params) * bkg(pts, params) / overall(pts, params), x=pts
    )

    return retval


def signal_weight_fcn(params: Tuple) -> Callable:
    """
    Find a function that gives us weights to project out the
    signal from the sWeighting

    """
    # Find the W matrix
    w_mat = w_matrix(params)

    # Find the alpha params
    alpha = solve(w_mat, [1, 0])

    # Construct the weights function
    return lambda x: (alpha[0] * sig(x, params) + alpha[1] * bkg(x, params)) / overall(
        x, params
    )


def sweights(dataframes: Iterable[pd.DataFrame], params: Tuple) -> Iterable[np.ndarray]:
    """
    Get a generator of sWeights from some mass fit parameters

    """
    # Find the weighting fcn
    fcn = signal_weight_fcn(params)

    for dataframe in dataframes:
        delta_m = dataframe["D* mass"] - dataframe["D0 mass"]
        yield fcn(delta_m)
