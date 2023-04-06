"""
Read the right branches, do the right stuff to the data
to get it into the right format for the classifier

Some branches we can just read directly; others require some calculation
(e.g. the sum of daughter pT) before they can be used

Most of the functions here are deprecated

"""
from typing import Tuple, Generator

import numpy as np
import pandas as pd


def refit_chi2(tree) -> np.ndarray:
    """
    The chi2 for the ReFit fit

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of ReFit values.
              Takes the first (best-fit) value from the jagged array

    """
    # Jagged array; take first (best-fit) value
    return tree["Dst_ReFit_chi2"].array()[:, 0]


def endvertex_chi2(tree) -> np.ndarray:
    """
    D0 end vertex chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of chi2 values.

    """
    return tree["D0_ENDVERTEX_CHI2"].array()


def orivx_chi2(tree) -> np.ndarray:
    """
    D0 origin vertex chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of chi2 values.

    """
    return tree["D0_ORIVX_CHI2"].array()


def slow_pi_prob_nn_pi(tree) -> np.ndarray:
    """
    Neural network pion probability for soft pion.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of pion probabilities

    """
    return tree["Dst_pi_ProbNNpi"].array()


def _pt(p_x: np.ndarray, p_y: np.ndarray) -> np.ndarray:
    """
    Transverse momentum

    """
    # pT is magnitude of px and py
    return np.sqrt(p_x**2 + p_y**2)


def _particle_names() -> Tuple[str, str, str, str]:
    """
    Particle naming convention for K 3pi

    """
    return ("Kplus", "piplus_0", "piplus_1", "piplus")


def d0_pt(tree) -> np.ndarray:
    """
    ReFit D0 pT, calculated from daughter refit momenta.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    # Get D0 pX and pY from the sum of daughter momenta
    p_x = np.sum(
        [tree[f"Dst_ReFit_D0_{x}_PX"].array()[:, 0] for x in _particle_names()], axis=0
    )
    p_y = np.sum(
        [tree[f"Dst_ReFit_D0_{x}_PY"].array()[:, 0] for x in _particle_names()], axis=0
    )

    return _pt(p_x, p_y)


def slow_pi_pt(tree) -> np.ndarray:
    """
    ReFit slow pi, calculated from slow pi momentum.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    return _pt(
        tree["Dst_ReFit_piplus_PX"].array()[:, 0],
        tree["Dst_ReFit_piplus_PY"].array()[:, 0],
    )


def slow_pi_ipchi2(tree) -> np.ndarray:
    """
    slow pi IP chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    return tree["Dst_pi_IPCHI2_OWNPV"].array()


def _refit_daughters_pt(tree) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transverse momentum of K, 3 pions

    """
    p_x: Generator = (
        tree[f"Dst_ReFit_D0_{x}_PX"].array()[:, 0] for x in _particle_names()
    )
    p_y: Generator = (
        tree[f"Dst_ReFit_D0_{x}_PY"].array()[:, 0] for x in _particle_names()
    )

    return tuple(_pt(x, y) for x, y in zip(p_x, p_y))


def pions_max_pt(tree) -> np.ndarray:
    """
    Max pT of daughter pions

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pions max pT

    """
    # Find the element-wise maximum of the pions pT
    # Only considering the pions so skip the first entry
    return np.amax(_refit_daughters_pt(tree)[1:], axis=0)


def pions_min_pt(tree) -> np.ndarray:
    """
    Min pT of daughter pions

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pions min pT

    """
    # Find the element-wise minimum of the pions pT
    # Only considering the pions so skip the first entry
    return np.amin(_refit_daughters_pt(tree)[1:], axis=0)


def pions_sum_pt(tree) -> np.ndarray:
    """
    Scalar sum of daughter pions pT

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pion pT sum

    """
    # Find the element-wise sum of the pions pT
    # Only considering the pions so skip the first entry
    return np.sum(_refit_daughters_pt(tree)[1:], axis=0)


def daughters_max_pt(tree) -> np.ndarray:
    """
    Max pT of daughter particles

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles max pT

    """
    # Find the element-wise maximum of the daughters pT
    return np.amax(_refit_daughters_pt(tree), axis=0)


def daughters_min_pt(tree) -> np.ndarray:
    """
    Min pT of daughter particles

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles min pT

    """
    # Find the element-wise minimum of the daughters pT
    return np.amin(_refit_daughters_pt(tree), axis=0)


def daughters_sum_pt(tree) -> np.ndarray:
    """
    Scalar sum of daughter particles pT

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles pT sum

    """
    # Find the element-wise sum of the daughters pT
    return np.sum(_refit_daughters_pt(tree), axis=0)


def slow_pi_pt_colname() -> str:
    """
    Column name for the slow pi pT

    """
    return r"$\pi_s$ $p_T$"


def add_slowpi_pt_col(dataframe: pd.DataFrame) -> None:
    """
    Add slow pion pT column to a dataframe in place

    """
    dataframe[slow_pi_pt_colname()] = _pt(
        dataframe["Dst_ReFit_piplus_PX"], dataframe["Dst_ReFit_piplus_PY"]
    )


def _training_var_names_and_functions() -> Tuple:
    """
    Both of them together, to make it easier to comment things out when I'm
    doing studies

    Returns (functions, names) as tuples

    """
    # Do it like this so we return two tuples but its easy to just comment
    # them out in the right way
    return tuple(
        zip(
            *(
                (refit_chi2, "Dst_ReFit_chi2"),
                (endvertex_chi2, "D0_ENDVERTEX_CHI2"),
                (orivx_chi2, "D0_ORIVX_CHI2"),
                (slow_pi_prob_nn_pi, "Dst_pi_ProbNNpi"),
                # (d0_pt, r"D0 $p_T$"),
                (slow_pi_pt, slow_pi_pt_colname()),
                (slow_pi_ipchi2, "Dst_pi_IPCHI2_OWNPV"),
                # (pions_max_pt, r"$3\pi$ max $p_T$"),
                # (pions_min_pt, r"$3\pi$ min $p_T$"),
                # (pions_sum_pt, r"$3\pi$ sum $p_T$"),
                # (daughters_max_pt, r"$K3\pi$ max $p_T$"),
                # (daughters_min_pt, r"$K3\pi$ min $p_T$"),
                # (daughters_sum_pt, r"$K3\pi$ sum $p_T$"),
            )
        )
    )


def training_var_functions() -> Tuple:
    """
    Returns a tuple of functions used for finding the training variables

    """
    return _training_var_names_and_functions()[0]


def training_var_names() -> Tuple:
    """
    Returns a tuple of the names of training variables

    """
    return _training_var_names_and_functions()[1]


def add_vars(dataframe: pd.DataFrame, tree, keep: np.ndarray) -> None:
    """
    Add branches containing the BDT cut training vars to the dataframe in place

    """
    for fcn, name in zip(*_training_var_names_and_functions()):
        dataframe[name] = fcn(tree)[keep]
