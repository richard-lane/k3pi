"""
Fit a parabola to the output of the Z scan

"""
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit


def fit_model(
    pts: np.ndarray,
    x_mean: float,
    y_mean: float,
    x_width: float,
    y_width: float,
    correlation: float,
) -> np.ndarray:
    """
    Model for 2d paraboloid

    :param pts: array of x and y pts, shape (2, N)
    :param means: (x mean, y mean)
    :param widths: (x width, y width)
    :param correlation: correlation between x and y

    """
    del_x = pts[0] - x_mean
    del_y = pts[1] - y_mean

    pre_factor = 0.5 / (1 - (correlation**2))
    x_term = del_x / x_width
    y_term = del_y / y_width
    cross_term = correlation * x_term * y_term

    return pre_factor * (x_term**2 + y_term**2 - 2 * cross_term)


def fit(
    chi2s: np.ndarray, re_z: np.ndarray, im_z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a 2d parabola to (some of) a chi2 scan

    :param chi2s: 2d array of chi2 values (not sqrted)
    :param re_z: real Z used in the scan
    :param im_z: imag Z used in the scan

    :returns: array of (x mean, y mean, x width, y width, correlation) fit values
    :returns: array of (x mean, y mean, x width, y width, correlation) errors

    """
    # Transform chi2 to delta chi2, and flatten the array
    delta_chi2 = chi2s - np.nanmin(chi2s)
    delta_chi2 = delta_chi2.ravel()

    # Toss away values that are too big - we only want to fit where the likelihood is parabolic
    max_chi2 = 9
    delta_chi2[delta_chi2 > max_chi2] = np.nan

    # Make 2d grids of Z points
    rez_2d, imz_2d = np.meshgrid(re_z, im_z)
    x_data = np.vstack((rez_2d.ravel(), imz_2d.ravel()))

    # Keep only the ones that are below the critical value
    keep = np.isfinite(delta_chi2)
    delta_chi2 = delta_chi2[keep]
    x_data = x_data[:, keep]

    # fit
    popt, pcov = curve_fit(
        fit_model,
        x_data,
        delta_chi2,
        p0=(0.75, -0.25, 0.1, 0.2, 0.0),
        bounds=((-1.5, -1.5, 0.0, 0.0, 0.0), (1.5, 1.5, np.inf, np.inf, 1.0)),
        maxfev=50000,
    )

    # Find errors
    errs = np.sqrt(np.diag(pcov))

    # Return parameters and errors
    return popt, errs


def parabola(params: np.ndarray, re_z: np.ndarray, im_z: np.ndarray) -> np.ndarray:
    """
    2d array of Z values given our params

    """
    x_x, y_y = np.meshgrid(re_z, im_z)
    pts = np.vstack((x_x.ravel(), y_y.ravel()))

    return fit_model(pts, *params).reshape((len(im_z), len(re_z)))
