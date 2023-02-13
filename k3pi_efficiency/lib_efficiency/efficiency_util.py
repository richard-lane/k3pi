"""
Utility functions for efficiency stuff

"""
import sys
import logging
import pathlib
from typing import Tuple, List
import numpy as np
import pandas as pd
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import definitions, util, get


def k_3pi(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the kaon and 3 pions as 4xN numpy arrays of (px, py, pz, E)

    """
    logging.warning(
        "Use the k3pi fcn in lib_data.util instead", exc_info=DeprecationWarning()
    )

    particles = [
        definitions.MOMENTUM_COLUMNS[0:4],
        definitions.MOMENTUM_COLUMNS[4:8],
        definitions.MOMENTUM_COLUMNS[8:12],
        definitions.MOMENTUM_COLUMNS[12:16],
    ]
    k, pi1, pi2, pi3 = (
        np.row_stack([dataframe[x] for x in labels]) for labels in particles
    )
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    return k, pi1, pi2, pi3


def points(
    k: np.ndarray, pi1: np.ndarray, pi2: np.ndarray, pi3: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    6d phsp + time points
    """
    return np.column_stack(
        (
            helicity_param(k, pi1, pi2, pi3),
            t,
        )
    )


def k_sign_cut(dataframe: pd.DataFrame, k_sign: str) -> pd.DataFrame:
    """
    Choose the right kaons - returns a copy

    """
    assert k_sign in {"k_minus", "k_plus", "both"}

    if k_sign == "both":
        return dataframe

    copy = dataframe.copy()

    k_ids = copy["K ID"].to_numpy()
    keep = k_ids < 0 if k_sign == "k_minus" else k_ids > 0

    print(f"k sign cut: keeping {np.sum(keep)} of {len(keep)}")

    return copy[keep]


def ampgen_df(decay_type: str, k_charge: str, train: bool) -> pd.DataFrame:
    """
    AmpGen dataframe

    :param decay_type: "dcs", "cf" or "false"
    :param k_charge: "k_plus", "k_minus" or "both"
    :param train: True, False or otherwise
                  Training events if True; testing events if False
                  Otherwise, both

    :returns: dataframe with 3-momenta flipped if required
              and a "K ID" column added

    """
    assert decay_type in {"dcs", "cf", "false"}
    assert k_charge in {"k_plus", "k_minus", "both"}

    # False sign looks like DCS in projections
    dataframe = get.ampgen("dcs" if decay_type == "false" else decay_type)

    if train is True:
        train_mask = dataframe["train"]
    elif train is False:
        train_mask = ~dataframe["train"]
    else:
        print("ampgen: using both test + train")
        train_mask = np.ones(len(dataframe), dtype=np.bool_)

    dataframe = dataframe[train_mask]

    # Add a K ID column to the dataframe
    dataframe["K ID"] = 321 * np.ones(len(dataframe))

    if k_charge == "k_plus":
        # Don't flip any momenta
        return dataframe

    if k_charge == "k_minus":
        # Flip all the momenta
        mask = np.ones(len(dataframe), dtype=np.bool_)

    elif k_charge == "both":
        # Flip half of the momenta randomly
        mask = np.random.random(len(dataframe)) < 0.5

    # Flip the right IDs if necessary
    dataframe.loc[mask, "K ID"] *= -1

    # Flip 3 momenta
    dataframe = util.flip_momenta(dataframe, mask)

    return dataframe


def pgun_df(decay_type: str, k_charge: str, train: bool) -> pd.DataFrame:
    """
    Particle gun dataframe

    """
    assert decay_type in {"dcs", "cf", "false"}
    assert k_charge in {"k_plus", "k_minus", "both"}

    if decay_type == "false":
        dataframe = get.false_sign()

    else:
        dataframe = get.particle_gun(decay_type, show_progress=True)

        # We only want to train on training data
        train_mask = dataframe["train"] if train else ~dataframe["train"]
        dataframe = dataframe[train_mask]

    # We may also only want to consider candidates with the same sign kaon
    dataframe = k_sign_cut(dataframe, k_charge)
    return dataframe


def _check_none_zero(list_of_arrays: List[np.ndarray]):
    """
    Check that none of the entries in a list of arrays is zero

    :raises AssertionError: if not

    """
    for array in list_of_arrays:
        assert np.count_nonzero(array) == len(array)


def _total_length(list_of_arrays: List[np.ndarray]):
    """
    Find the total length of a list of arrays

    :returns: sum of lengths of arrays in a list

    """
    retval = 0
    for array in list_of_arrays:
        retval += len(array)

    return retval


def scale_weights(
    rs_wts: List[np.ndarray], ws_wts: List[np.ndarray], avg_eff_ratio: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Scale lists of RS and WS weights to have the right dcs/cf ratio
    Does not scale the CF weights

    :param rs_wts: list of arrays of efficiency weights for a RS sample
    :param ws_wts: list of arrays of efficiency weights for a WS sample
    :param avg_eff_ratio: average of dcs/cf absolute efficiencies. DCS_AVG / CF_AVG

    :returns: scaled list of RS weights
    :returns: scaled list of WS weights

    """
    # Check that there are no zero weight in our sample,
    # since this is likely to break things we think
    _check_none_zero(rs_wts)
    _check_none_zero(ws_wts)

    # Find the number of each type of weight
    # This is the number of (reconstructed) events
    n_rs = _total_length(rs_wts)
    n_ws = _total_length(ws_wts)

    # Find the average of each type of weight
    rs_avg_wt = np.sum(rs_wts) / n_rs
    ws_avg_wt = np.sum(ws_wts) / n_ws

    # We want sum of WS wts to equal N WS generated evts,
    # which is equivalent to N WS reco evts / avg eff
    # Scaling factor for WS works out to this, leaving
    # RS unscaled
    scale_factor = n_ws * rs_avg_wt / (n_rs * ws_avg_wt * avg_eff_ratio)
    print(f"Scaling weights by {scale_factor}")
    scaled_ws_wts = [arr * scale_factor for arr in ws_wts]

    return rs_wts, scaled_ws_wts
