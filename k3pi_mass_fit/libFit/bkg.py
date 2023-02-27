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


def create_dump(
    counts: np.ndarray,
    errors: np.ndarray,
    n_bins: int,
    sign: str,
    *,
    bdt_cut: bool,
    efficiency: bool,
) -> None:
    """
    Create an array of estimated background counts from a pickle dump

    """
    assert len(counts) == n_bins
    assert len(counts) == len(errors)

    path = dump_path(n_bins, sign, bdt_cut=bdt_cut, efficiency=efficiency)

    with open(path, "wb") as bkg_f:
        print(f"dumping {path}")
        pickle.dump((counts, errors), bkg_f)


def pdf(
    bins: np.ndarray, year: str, magnetisation: str, sign: str, *, bdt_cut: bool
) -> Callable:
    """
    Get a function that returns normalised probability density from
    an estimated background histogram

    :param n_bins: how many mass bins
    :param sign: dcs or cf
    :param bdt_cut: whether to do the BDT cut
    :param efficiency: whether to do the efficiency correction

    :returns: a function that takes a mass difference and returns normalised
              probability density

    """
    # Get the histogram
    counts, _ = get_counts(year, magnetisation, sign, bins, bdt_cut=bdt_cut)

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
