"""
Decay time fitters

"""
from typing import Tuple
import numpy as np
from iminuit import Minuit

from . import util, models


def weighted_mean(points: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Weighted mean and its error of some points

    :param points: values
    :param errors: errors on the values
    :returns: the mean
    :returns: error on the mean

    """
    weights = 1 / errors**2

    return np.average(points, weights=weights), 1 / np.sum(weights)


def no_mixing(ratio: np.ndarray, err: np.ndarray) -> Tuple[float, float]:
    r"""
    Find the best value that describes our dataset if we assume no mixing;
    i.e. that the ratio is described by r_D^2.

    We don't actually need to do a fit here - we can find the optimal
    \chi^2 value analytically - it's just the weighted mean

    :param ratio: binned ratio of DCS/CF decay times
    :param err: error on the ratio

    :returns: the value of r_D that best describes the data
    :returns: the error on r_D

    """
    assert len(ratio) == len(err)

    mean, err_on_mean = weighted_mean(ratio, err)

    amp_ratio = np.sqrt(mean)
    amp_ratio_err = 0.5 * err_on_mean / np.sqrt(mean)

    return amp_ratio, amp_ratio_err


def no_constraints(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    initial_guess: util.MixingParams,
) -> Minuit:
    """
    Fit WS/RS decay time ratio with no constraints on parameters

    Uses models.no_constraints as the model.

    :param ratio: ratio of WS/RS decay times in each time bin
    :param errs: error on the ratio of WS/RS decay times in each time bin
    :param bins: bins used when calculating the ratio.
                 Should contain each left bin edge, plus the rightmost.
    :param initial_guess: inital guess at the parameters when fitting.

    :returns: Minuit fitter after the fit


    """
    assert len(ratio) == len(bins) - 1

    chi2 = models.NoConstraints(ratio, errs, bins)

    minimiser = Minuit(chi2, **initial_guess._asdict())

    minimiser.migrad()

    return minimiser


def constraints(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    initial_guess: util.ConstraintParams,
    x_y_widths: Tuple[float, float],
    x_y_correlation: float,
) -> Minuit:
    """
    Fit WS/RS decay time ratio
    with Gaussian constraints on the mixing parameters.

    Uses models.constraints as the model.
    Gaussian constraint uses the initial values of x, y as the
    mean and the provided widths/correlation.

    :param ratio: ratio of WS/RS decay times in each time bin
    :param errs: error on the ratio of WS/RS decay times in each time bin
    :param bins: bins used when calculating the ratio.
                 Should contain each left bin edge, plus the rightmost.
    :param initial_guess: inital guess at the parameters when fitting.
    :param x_y_widths: tuple of x, y widths to use in the Gaussian constraint
    :param x_y_correlation: measured correlation between x and y

    :returns: Minuit fitter after the fit


    """
    assert len(ratio) == len(bins) - 1

    chi2 = models.Constraints(
        ratio,
        errs,
        bins,
        (initial_guess.x, initial_guess.y),
        x_y_widths,
        x_y_correlation,
    )

    minimiser = Minuit(chi2, **initial_guess._asdict())

    minimiser.migrad()

    return minimiser


def scan_fit(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    initial_guess: util.ConstraintParams,
    x_y_widths: Tuple[float, float],
    x_y_correlation: float,
) -> Minuit:
    """
    Fit WS/RS decay time ratio
    with Gaussian constraints on the mixing parameters
    and fixed Z

    Uses models.scan as the model.
    Gaussian constraint uses the initial values of x, y as the
    mean and the provided widths/correlation.

    :param ratio: ratio of WS/RS decay times in each time bin
    :param errs: error on the ratio of WS/RS decay times in each time bin
    :param bins: bins used when calculating the ratio.
                 Should contain each left bin edge, plus the rightmost.
    :param initial_guess: inital guess at the parameters when fitting.
    :param x_y_widths: tuple of x, y widths to use in the Gaussian constraint
    :param x_y_correlation: measured correlation between x and y

    :returns: Minuit fitter after the fit


    """
    assert len(ratio) == len(bins) - 1

    chi2 = models.Scan(
        ratio,
        errs,
        bins,
        (initial_guess.x, initial_guess.y),
        x_y_widths,
        x_y_correlation,
        (initial_guess.re_z, initial_guess.im_z),
    )

    # Don't need to pass the whole initial guess to minuit
    # since we aren't varying Z
    params = initial_guess._asdict()
    params.pop("re_z")
    params.pop("im_z")
    minimiser = Minuit(chi2, **params)

    minimiser.migrad()

    return minimiser


def combined_fit(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    initial_guess: util.ScanParams,
    x_y_widths: Tuple[float, float],
    x_y_correlation: float,
    bin_number: int,
) -> Minuit:
    """
    Fit WS/RS decay time ratio
    with Gaussian constraints on the mixing parameters
    and fixed Z, and a constraint from the charm threshhold

    Uses models.CharmThreshholdScan as the model.
    Gaussian constraint uses the initial values of x, y as the
    mean and the provided widths/correlation.

    :param ratio: ratio of WS/RS decay times in each time bin
    :param errs: error on the ratio of WS/RS decay times in each time bin
    :param bins: bins used when calculating the ratio.
                 Should contain each left bin edge, plus the rightmost.
    :param initial_guess: inital guess at the parameters when fitting.
    :param x_y_widths: tuple of x, y widths to use in the Gaussian constraint
    :param x_y_correlation: measured correlation between x and y
    :param bin_number: phase space bin number

    :returns: Minuit fitter after the fit


    """
    assert len(ratio) == len(bins) - 1

    chi2 = models.CharmThreshholdScan(
        ratio,
        errs,
        bins,
        (initial_guess.x, initial_guess.y),
        x_y_widths,
        x_y_correlation,
        (initial_guess.re_z, initial_guess.im_z),
        bin_number,
    )

    # Don't need to pass the whole initial guess to minuit
    # since we aren't varying Z
    params = initial_guess._asdict()
    params.pop("re_z")
    params.pop("im_z")
    minimiser = Minuit(chi2, **params)

    minimiser.migrad()

    return minimiser
