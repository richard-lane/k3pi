"""
Utility functions that may be useful

"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import uproot

from . import definitions


def convert_to_kplus(
    k: np.ndarray, pi1: np.ndarray, pi2: np.ndarray, pi3: np.ndarray, k_id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flip 3 momenta of k 3pi by using the kaon IDs

    """
    to_flip = k_id < 0

    k[1:][:, to_flip] *= -1
    pi1[1:][:, to_flip] *= -1
    pi2[1:][:, to_flip] *= -1
    pi3[1:][:, to_flip] *= -1

    return k, pi1, pi2, pi3


def flip_momenta(dataframe: pd.DataFrame, to_flip=None) -> pd.DataFrame:
    """
    In some cases, we may want to only consider one type of decay (e.g. D0 -> K+3pi
    instead of both D0->K+3pi and Dbar0->K-3pi). In some of these cases, we may want
    to convert the K- type momenta to what it would be had the decay been to a K+ -
    i.e. we want to flip the 3 momentum by multiplying by -1.

    Returns a copy

    :param to_flip: mask of candidates to flip. If not provided, then flips candidates
                    where df["K_ID"] < 0

    """

    flip_columns = [
        *definitions.MOMENTUM_COLUMNS[0:3],
        *definitions.MOMENTUM_COLUMNS[4:7],
        *definitions.MOMENTUM_COLUMNS[8:11],
        *definitions.MOMENTUM_COLUMNS[12:15],
    ]
    to_flip = dataframe["K ID"].to_numpy() < 0 if to_flip is None else to_flip
    print(f"flipping {np.sum(to_flip)} momenta of {len(to_flip)}")

    df_copy = dataframe.copy()
    for col in flip_columns:
        df_copy[col][to_flip] = -1 * dataframe[col][to_flip]

    return df_copy


def invariant_masses(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray
) -> np.ndarray:
    """
    Find the invariant masses of a collection of particles represented by their kinematic data

    :param px: particle x momenta
    :param py: particle y momenta
    :param pz: particle z momenta
    :param energy: particle energies
    :returns: array of particle invariant masses

    """
    return np.sqrt(energy ** 2 - px ** 2 - py ** 2 - pz ** 2)


def momentum_order(
    k: np.ndarray, pi1: np.ndarray, pi2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order two pions based on the invariant mass M(Kpi)

    :param k: kaon parameters (px, py, pz, E)
    :param pi1: pion parameters (px, py, pz, E)
    :param pi2: pion parameters (px, py, pz, E)

    :returns: (lower_mass_pion, higher_mass_pion) as their pion parameters.
              Returns copies of the original arguments

    """
    new_pi1, new_pi2 = np.zeros((4, len(k.T))), np.zeros((4, len(k.T)))

    # Find invariant masses
    m1 = invariant_masses(*np.add(k, pi1))
    m2 = invariant_masses(*np.add(k, pi2))

    # Create bool mask of telling us which pion has the higher mass
    mask = m1 > m2  # pi1[mask] selects high-mass pions
    inv_mask = ~mask  # pi2[inv_mask] selects high-mass pions

    # Fill new arrs
    new_pi1[:, inv_mask] = pi1[:, inv_mask]
    new_pi1[:, mask] = pi2[:, mask]

    new_pi2[:, mask] = pi1[:, mask]
    new_pi2[:, inv_mask] = pi2[:, inv_mask]

    return new_pi1, new_pi2


def add_train_column(
    gen: np.random.Generator, dataframe: pd.DataFrame, train_fraction: float = 0.5
) -> None:
    """
    Use a random number generator to add a boolean column for training to a
    dataframe in place

    :param gen: RNG
    :param dataframe: dataframe to add a boolean column titled "train" to.
                      Adds this column in place
    :param train_fraction: approx proportion of True in the column

    """
    assert (0.0 <= train_fraction) and (train_fraction <= 1.0)

    dataframe["train"] = gen.random(len(dataframe)) < train_fraction


def luminosity(filepath: str) -> float:
    """
    Get the luminosity from a file

    """
    with uproot.open(filepath) as root_file:
        return np.sum(
            root_file["GetIntegratedLuminosity/LumiTuple"][
                "IntegratedLuminosity"
            ].array()
        )


def total_luminosity(files: List[str]) -> float:
    """
    Get the total luminosity for a collection of files

    """
    total_lumi = 0
    for path in tqdm(files):
        total_lumi += luminosity(path)

    return total_lumi
