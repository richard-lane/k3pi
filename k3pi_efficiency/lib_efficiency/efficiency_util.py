"""
Utility functions for efficiency stuff

"""
import sys
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

    if k_charge == "k_plus":
        # Don't flip any momenta
        return dataframe

    if k_charge == "k_minus":
        # Flip all the momenta
        mask = np.ones(len(dataframe), dtype=np.bool_)

    elif k_charge == "both":
        # Flip half of the momenta randomly
        mask = np.random.random(len(dataframe)) < 0.5

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


def scale_weights(
    rs_wts: List[np.ndarray], ws_wts: List[np.ndarray], dcs_cf_ratio: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Scale lists of RS and WS weights to have the right dcs/cf ratio

    Does not scale the CF weights

    :param rs_wts: list of arrays of efficiency weights for a RS sample
    :param ws_wts: list of arrays of efficiency weights for a WS sample
    :param dcs_cf_ratio: desired average of dcs/cf weights. DCS_AVG / CF_AVG

    :returns: scaled list of RS weights
    :returns: scaled list of WS weights

    """
    # Find the sum of each type of weight
    rs_sum = np.sum(rs_wts)
    ws_sum = np.sum(ws_wts)

    # Scale them
    scale_factor = dcs_cf_ratio * rs_sum / ws_sum
    scaled_ws_wts = [arr * scale_factor for arr in ws_wts]

    return rs_wts, scaled_ws_wts
