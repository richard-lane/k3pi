"""
Functions for performing straight cuts

Each function returns a boolean mask of which events to keep

"""
import numpy as np
from uproot.exceptions import KeyInFileError


def d0_mass(tree) -> np.ndarray:
    """
    The best fit mass of the D0 after ReFit

    """
    # Jagged array; take the first (best-fit) value for each
    return tree["Dst_ReFit_D0_M"].array(library="ak")[:, 0]


def dst_mass(tree) -> np.ndarray:
    """
    Best fit mass of D* after ReFit

    """
    return tree["Dst_ReFit_M"].array(library="ak")[:, 0]


# ==== Individual, branch level cuts
def _d0_mass_keep(tree) -> np.ndarray:
    """
    Keep events near the nominal D0 mass

    """
    min_mass, max_mass = 1840.83, 1888.83
    d0_m = d0_mass(tree)

    return (min_mass < d0_m) & (d0_m < max_mass)


def _delta_m_keep(tree) -> np.ndarray:
    """
    Keep events where the D* - D0 mass difference is near the nominal pi mass

    """
    min_mass, max_mass = 139.3, 152.0

    # Jagged array - take the first (best fit) value for the D* masses
    delta_m = dst_mass(tree) - d0_mass(tree)

    return (min_mass < delta_m) & (delta_m < max_mass)


def _ipchi2_keep(tree) -> np.ndarray:
    """
    Keep events where the IPCHI2 for the D0 is small

    """
    return tree["D0_IPCHI2_OWNPV"].array(library="np") < 9.0


def _l0_keep(tree) -> np.ndarray:
    """
    Keep any events where the global L0 trigger is TIS or hadron trigger is TOS

    """
    return (tree["D0_L0HadronDecision_TOS"].array(library="np") == 1) | (
        tree["Dst_L0Global_TIS"].array(library="np") == 1
    )


def _hlt_keep(tree) -> np.ndarray:
    """
    Keep any events where either the 1 or 2 track HLT1 decision for the D* is TOS

    Use this instead of the D0 as there's no D0 info for the
    particle gun, and we want the cuts to be consistent
    between all types of data


    """
    # TODO neaten
    try:
        # Data, MC uses this naming convention
        keep_mask = (tree["Dst_Hlt1TrackMVADecision_TOS"].array(library="np") == 1) | (
            tree["Dst_Hlt1TwoTrackMVADecision_TOS"].array(library="np") == 1
        )
    except KeyInFileError:
        # Particle gun uses this one
        keep_mask = (
            tree["Dplus_Hlt1TrackMVADecision_TOS"].array(library="np") == 1
        ) | (tree["Dplus_Hlt1TwoTrackMVADecision_TOS"].array(library="np") == 1)

    return keep_mask


def _bkgcat(tree) -> np.ndarray:
    """
    Keep signal events only by cutting on the D0 and D* background category

    """
    return (tree["D0_BKGCAT"].array(library="np") == 0) & (
        tree["Dst_BKGCAT"].array(library="np") == 0
    )


def _pid_keep(tree) -> np.ndarray:
    """
    Keep only events with good enough slowpi probNNpi and probNNk

    """
    # Find kaon and pion probabilities for each daughter,
    # cut on some combination of these
    keep = []
    for i in range(4):
        pi_prob = tree[f"D0_P{i}_ProbNNpi"].array(library="np")
        k_prob = tree[f"D0_P{i}_ProbNNk"].array(library="np")

        keep.append((pi_prob * (1 - k_prob)) > 0.6)

    # We want the kaon selection criterion to be the opposite of the pions
    keep[0] = ~keep[0]

    # Keep only events which pass all of the PID criteria
    return np.logical_and.reduce(keep)


# ==== Combinations of cuts
def _sanity_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after sanity cuts (D/D* mass, ipchi2)

    """
    return np.logical_and.reduce(
        [
            fcn(tree)
            for fcn in (
                _d0_mass_keep,
                _delta_m_keep,
                _ipchi2_keep,
            )
        ]
    )


def _trigger_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after HLT cuts

    """
    return np.logical_and(_l0_keep(tree), _hlt_keep(tree))


def data_keep(tree) -> np.ndarray:
    """
    Which events to keep in real data

    """
    return _sanity_keep(tree) & _trigger_keep(tree) & _pid_keep(tree)


def uppermass_keep(tree) -> np.ndarray:
    """
    Which events to keep for upper mass sideband

    Same as data but without the cuts on D0 mass/delta M

    """
    return _ipchi2_keep(tree) & _trigger_keep(tree) & _pid_keep(tree)


def simulation_keep(tree) -> np.ndarray:
    """
    Which events to keep in simulated data

    The same as data cuts, but with BKGCAT cut (truth matching)

    """
    return _sanity_keep(tree) & _trigger_keep(tree) & _bkgcat(tree)


def pgun_keep(data_tree, hlt_tree) -> np.ndarray:
    """
    Which events to keep for particle gun

    Particle gun is special - there's no L0 info
    (TIS doesn't make sense, TOS info isn't in the tree)
    so we just do HLT cuts instead of HLT and L0 trigger

    """
    return _sanity_keep(data_tree) & _hlt_keep(hlt_tree) & _bkgcat(tree)
