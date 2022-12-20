"""
Tools for estimating the background by combining our
K3pi with a random slow pion

"""
import pathlib
import pickle
from typing import Tuple
import numpy as np


def dump_path(
    n_bins: int, sign: str, *, bdt_cut: bool, efficiency: bool
) -> pathlib.Path:
    """
    Location of the background pickle dump

    """
    assert sign in {"dcs", "cf"}

    suffix = ""
    if bdt_cut:
        suffix += "_cut"
    if efficiency:
        assert bdt_cut
        suffix += "_eff"

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / f"bkg_dump_{sign}_{n_bins}_bins{suffix}.pkl"
    )


def get_dump(
    n_bins: int, sign: str, *, bdt_cut: bool, efficiency: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get an array of estimated background counts from a pickle dump

    :param n_bins: number of mass bins
    :param sign: dcs or cf
    :param bdt_cut: whether to do the BDT cut before estimating background
    :param efficiency: whether to do the efficiency correction before estimating background

    :returns: array of counts
    :returns: array of errors

    """
    path = dump_path(n_bins, sign, bdt_cut=bdt_cut, efficiency=efficiency)

    with open(path, "rb") as bkg_f:
        print(f"loading {path}")
        return pickle.load(bkg_f)


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
