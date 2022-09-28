"""
Read the right branches, do the right stuff to the data
to get it into the right format for the classifier

Some branches we can just read directly; others require some calculation (e.g. the sum of daughter pT) before they can be used

"""
from typing import Tuple, Generator, Callable

import numpy as np


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
    return np.sqrt(p_x ** 2 + p_y ** 2)


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


def decay_times(tree) -> np.ndarray:
    """
    Decay time of D0 meson, in ps

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of decay times

    """
    # Jagged array; take first (best-fit) value
    # Divide by 0.3 to get from c*tau to ps
    return tree["Dst_ReFit_D0_ctau"].array()[:, 0] / 0.3


def training_var_functions() -> Tuple:
    """
    Returns a tuple of functions used for finding the training variables

    """
    return (
        endvertex_chi2,
        orivx_chi2,
        slow_pi_prob_nn_pi,
        d0_pt,
        slow_pi_pt,
        slow_pi_ipchi2,
        pions_max_pt,
        pions_min_pt,
        pions_sum_pt,
        daughters_max_pt,
        daughters_min_pt,
        daughters_sum_pt,
    )


def training_var_names() -> Tuple:
    """
    Returns a tuple of the names of training variables

    Contains matplotlib-style LaTeX formatting: suitable for using as plot labels, etc.

    """
    return (
        r"D0 End Vtx $\chi^2$",
        r"D0 Origin Vtx $\chi^2$",
        r"$\pi_s$ ProbNN$\pi$",
        r"D0 $p_T$",
        r"$\pi_s$ $p_T$",
        r"$\pi_s$ IP$\chi^2$",
        r"$3\pi$ max $p_T$",
        r"$3\pi$ min $p_T$",
        r"$3\pi$ sum $p_T$",
        r"$K3\pi$ max $p_T$",
        r"$K3\pi$ min $p_T$",
        r"$K3\pi$ sum $p_T$",
    )


def points(tree) -> np.ndarray:
    """
    Data used to train the classifier; shape (Nx13)

    """
    return np.column_stack([f(tree) for f in training_var_functions()])


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

    :returns: (lower_mass_pion, higher_mass_pion) as their pion parameters. Returns copies of the original arguments

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
