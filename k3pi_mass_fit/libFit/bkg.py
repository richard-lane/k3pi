"""
Tools for estimating the background by combining our
K3pi with a random slow pion

"""
import sys
import glob
import pathlib
import pickle
from typing import Tuple, Callable, Iterable

import numpy as np
from scipy.stats import norm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import stats
from lib_data.util import check_year_mag_sign


def dump_dir(
    year: str, magnetisation: str, sign: str, *, bdt_cut: bool
) -> pathlib.Path:
    """
    Where the dump arrays are stored

    :param year: data taking year
    :param magnetisation: "magdown" or "magup"
    :param sign: "cf" or "dcs"
    :param bdt_cut: whether to bdt cut

    """
    check_year_mag_sign(year, magnetisation, sign)

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "bkg_dumps"
        / f"{year}_{magnetisation}_{sign}{'_cut' if bdt_cut else ''}/"
    )


def get_dumps(
    year: str,
    magnetisation: str,
    sign: str,
    *,
    bdt_cut: bool,
) -> Iterable[np.ndarray]:
    """
    Generator of arrays

    """
    dirname = dump_dir(year, magnetisation, sign, bdt_cut=bdt_cut)

    for path in glob.glob(str(dirname / "*")):
        with open(path, "rb") as dump_f:
            yield pickle.load(dump_f)


def get_counts(
    year: str,
    magnetisation: str,
    sign: str,
    bins: np.ndarray,
    *,
    bdt_cut: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the bkg counts and errors in each bin

    """
    return stats.counts_generator(
        get_dumps(year, magnetisation, sign, bdt_cut=bdt_cut), bins
    )


def _extended_bins(bins: np.ndarray, n_low: int, n_high: int) -> np.ndarray:
    """
    Bins with a few higher for under/overflow, and infs on each end

    """
    low_w = bins[1] - bins[0]
    high_w = bins[-1] - bins[-2]

    return np.array(
        (
            -np.inf,
            *[bins[0] - i * low_w for i in range(n_low, 0, -1)],
            *bins,
            *[bins[-1] + i * high_w for i in range(1, n_high + 1)],
            np.inf,
        )
    )


def _integral(fcn: Callable, domain: Tuple[float, float]) -> float:
    """
    Approx integral over a region

    """
    pts = np.linspace(*domain, 10000)
    vals = fcn(pts)

    return np.sum(0.5 * (vals[1:] + vals[:-1]) * (pts[1] - pts[0]))


def pdf(
    bins: np.ndarray, year: str, magnetisation: str, sign: str, *, bdt_cut: bool
) -> Callable:
    """
    Get a function that returns normalised probability density from
    an estimated background histogram

    :param bins: the bins to use
    :param year: for finding the right dump
    :param magnetisation: for finding the right dump
    :param sign: for finding the right dump
    :param bdt_cut: whether to do the BDT cut

    :returns: a function that takes a mass difference and returns normalised
              probability density
              Fcn is defined from bins[0] to bins[-1], and is
              normalised over this region

    """
    # Bins with an underflow and a few bins for overflow
    # The first/last bins are true under/overflow;
    # the last few bins are used for smoothing the KDE
    extended_bins = _extended_bins(bins, n_low=8, n_high=15)
    print(f"extended bins from {extended_bins[1]} to {extended_bins[-2]}")

    # Get the histogram
    ext_counts, _ = get_counts(
        year, magnetisation, sign, extended_bins, bdt_cut=bdt_cut
    )

    assert np.sum(ext_counts), "Empty histogram - have you created the dump?"

    print(f"{ext_counts[0]} underflow; {ext_counts[-1]} overflow")

    # Get rid of under and overflow
    ext_counts = ext_counts[1:-1]
    extended_bins = extended_bins[1:-1]

    # Get bin centres and widths
    ext_centres = (extended_bins[1:] + extended_bins[:-1]) / 2
    ext_widths = extended_bins[1:] - extended_bins[:-1]

    # Get a gaussian of the right width in each one
    gaussians = [
        norm(loc=centre, scale=4 * width)
        for (centre, width) in zip(ext_centres, ext_widths)
    ]

    # Build a function that get the sum of gaussians
    def fcn(points: np.ndarray) -> np.ndarray:
        """sum of gaussians, not normalised"""
        retval = np.zeros_like(points)

        for gaussian, count in zip(gaussians, ext_counts):
            retval += count * gaussian.pdf(points)

        return retval

    # Get the integral over the original bin range
    integral = _integral(fcn, (bins[0], bins[-1]))

    # Scale the fcn by its integral
    def scaled_fcn(points: np.ndarray) -> np.ndarray:
        """scaled sum of gaussians"""
        return fcn(points) / integral

    # Return this fcn
    return scaled_fcn
