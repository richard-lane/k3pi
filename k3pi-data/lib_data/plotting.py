"""
Useful things for plotting

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
from lib_efficiency.metrics import _counts


def _centres_widths(bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin centres and widths

    """
    return (bins[1:] + bins[:-1]) / 2, (bins[1:] - bins[:-1]) / 2


def _norm_hist(
    points: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalised weighted histogram and error

    """
    num = np.sum(weights)
    count, err = _counts(points, weights, bins)

    return count / num, err / num


def hist(
    axis: plt.Axes,
    bins: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray = None,
    **plot_kw,
) -> None:
    """
    Plot a normalised histogram on an axis

    """
    if weights is None:
        weights = np.ones_like(points)

    count, err = _norm_hist(points, weights, bins)

    centres, widths = _centres_widths(bins)
    axis.errorbar(centres, count, xerr=widths, yerr=err, **plot_kw)


def pull(
    axis: plt.Axes,
    bins: np.ndarray,
    points: Tuple[np.ndarray, np.ndarray],
    weights: Tuple[np.ndarray, np.ndarray],
    **plot_kw,
) -> None:
    """
    Plot the pull between two possibly weighted histograms

    """
    counts = []
    errs = []
    for point, weight in zip(points, weights):
        if weight is None:
            weight = np.ones_like(point)

        count, err = _norm_hist(point, weight, bins)
        counts.append(count)
        errs.append(err)

    diff = counts[0] - counts[1]
    err = np.sqrt(errs[0] ** 2 + errs[1] ** 2)

    centres, widths = _centres_widths(bins)
    axis.errorbar(centres, diff, xerr=widths, yerr=err, **plot_kw)
