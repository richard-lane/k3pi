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

from . import definitions

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
        / f"{year}_{magnetisation}_{sign}/"
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
    # Bins with an under and overflow
    extended_bins = (-np.inf, *bins, np.inf)

    # Get the histogram
    counts, _ = get_counts(year, magnetisation, sign, extended_bins, bdt_cut=bdt_cut)

    assert np.sum(counts), "Empty histogram - have you created the dump?"

    print(f"{counts[0]} underflow (<{bins[0]}); {counts[-1]} overflow (>{bins[-1]})")

    # Get rid of under and overflow
    counts = counts[1:-1]

    # Normalise counts
    widths = bins[1:] - bins[:-1]
    counts /= widths * np.sum(counts)

    def fcn(point: float):
        """histogram -> pdf"""
        # Bin the point
        index = np.digitize(point, bins) - 1

        assert np.all(index > -1)
        assert np.all(index < len(bins) - 1)

        # Return the counts at that point
        return counts[index]

    return fcn
