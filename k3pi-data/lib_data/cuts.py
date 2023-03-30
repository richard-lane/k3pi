"""
Functions for performing straight cuts

Each function returns a boolean mask of which events to keep

"""
import sys
import pathlib
import numpy as np
import pandas as pd

from . import util, definitions

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from libFit import pdfs

ANGLE_CUT_DEG = 0.03
MAX_IPCHI2 = 9.0
MAX_LIFETIMES = 8.0


def d0_mass(tree) -> np.ndarray:
    """
    The best fit mass of the D0 after ReFit

    """
    # Jagged array; take the first (best-fit) value for each
    return tree["Dst_ReFit_D0_M"].array(library="ak")[:, 0].to_numpy()


def dst_mass(tree) -> np.ndarray:
    """
    Best fit mass of D* after ReFit

    """
    return tree["Dst_ReFit_M"].array(library="ak")[:, 0].to_numpy()


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
    min_mass, max_mass = pdfs.domain()

    # Jagged array - take the first (best fit) value for the D* masses
    delta_m = dst_mass(tree) - d0_mass(tree)

    return (min_mass < delta_m) & (delta_m < max_mass)


def _time_keep(tree) -> np.ndarray:
    """
    Refit time < the chosen number of lifetimes

    """
    return (
        tree["Dst_ReFit_D0_ctau"].array(library="ak")[:, 0]
        / (0.3 * definitions.D0_LIFETIME_PS)
        < MAX_LIFETIMES
    )


def _ghost_keep(tree) -> np.ndarray:
    """
    Keep events with small ghost probability

    """
    return tree["Dst_pi_ProbNNghost"].array(library="np") < 0.1


def _angle_keep(tree) -> np.ndarray:
    """
    Keep events with large angles between hadrons

    """
    # Start by assuming we will keep everything
    retval = np.ones(tree.num_entries, dtype=bool)

    # Find 3-momenta of each species
    suffices = "PX", "PY", "PZ"

    momenta = tuple(
        np.row_stack(
            [
                tree[f"{prefix}_{suffix}"].array(library="ak")[:, 0].to_numpy()
                for suffix in suffices
            ]
        )
        for prefix in definitions.DATA_BRANCH_PREFICES
    )

    # Find angle between each one
    # Require that all angles > minimum value
    n_particles = 5
    for i in range(n_particles):
        for j in range(n_particles):
            if i == j:
                continue
            angle = util.relative_angle(momenta[i], momenta[j])
            retval &= angle > ANGLE_CUT_DEG

    return retval


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
    Keep any events where either the 1 or 2 track HLT1 decision for the D0 is TOS

    """
    return (tree["D0_Hlt1TrackMVADecision_TOS"].array(library="np") == 1) | (
        tree["D0_Hlt1TwoTrackMVADecision_TOS"].array(library="np") == 1
    )


def _hlt_keep_pgun(tree) -> np.ndarray:
    """
    Keep any events where either the 1 or 2 track HLT1 decision for the D0 is TOS

    Different branch naming convention for particle gun means we need this separate function

    """
    # Particle gun D0 HLT branch is named Dplus
    # ask Nathan Jurik if you care about why this is
    return (tree["Dplus_Hlt1TrackMVADecision_TOS"].array(library="np") == 1) | (
        tree["Dplus_Hlt1TwoTrackMVADecision_TOS"].array(library="np") == 1
    )


def _bkgcat(tree) -> np.ndarray:
    """
    Keep signal events only by cutting on the D0 and D* background category

    """
    return (tree["D0_BKGCAT"].array(library="np") == 0) & (
        tree["Dst_BKGCAT"].array(library="np") == 0
    )


def _pid_keep(tree) -> np.ndarray:
    """
    Keep only events with good enough daughter PIDK

    """
    # Find kaon and pion probabilities for each daughter,
    # cut on some combination of these
    retval = np.ones(tree.num_entries, dtype=bool)

    # Kaon
    retval &= tree["D0_P0_PIDK"].array(library="np") > 8

    # Opposite sign pion
    retval &= tree["D0_P1_PIDK"].array(library="np") < 0
    retval &= tree["D0_P3_PIDK"].array(library="np") < 0

    return retval


def _cands_keep(tree) -> np.ndarray:
    """
    Keep only events with nCandidate == 0 because it's easier than
    choosing them randomly

    """
    arr = tree["nCandidate"].array(library="np") == 0
    print(np.sum(arr), len(arr), sep="/")

    return arr


def _trueorigvtx_keep(tree) -> np.ndarray:
    """
    Truth matching cut enforcing that Dst true orig vtx is not 0

    For the particle gun

    """
    return tree["Dst_TRUEORIGINVERTEX_X"].array(library="np") != 0.0


# ==== Combinations of cuts
def _sanity_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after sanity cuts (just D/D* mass)

    """
    return np.logical_and.reduce(
        [
            fcn(tree)
            for fcn in (
                _d0_mass_keep,
                _delta_m_keep,
            )
        ]
    )


def _trigger_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after HLT cuts

    """
    return np.logical_and(_l0_keep(tree), _hlt_keep(tree))


def uppermass_keep(tree) -> np.ndarray:
    """
    Which events to keep for upper mass sideband

    PID, trigger and cuts for rejecting multiple candidates

    """
    return _trigger_keep(tree) & _pid_keep(tree) & _cands_keep(tree)


def data_keep(tree) -> np.ndarray:
    """
    Which events to keep in real data

    Same as uppermass but with extra cuts on D0 mass and delta M

    """
    return uppermass_keep(tree) & _sanity_keep(tree)


def simulation_keep(tree) -> np.ndarray:
    """
    Which events to keep in simulated data

    The same as data cuts, but with BKGCAT cut (truth matching) and D0 ipchi2 cut

    """
    return _ipchi2_keep(tree) & _sanity_keep(tree) & _trigger_keep(tree) & _bkgcat(tree)


def pgun_keep(data_tree, hlt_tree) -> np.ndarray:
    """
    Which events to keep for particle gun

    Particle gun is special - there's no L0 info
    (TIS doesn't make sense, TOS info isn't in the tree)
    so we just do HLT cuts instead of HLT and L0 trigger

    """
    return (
        _ipchi2_keep(data_tree)
        & _sanity_keep(data_tree)
        & _hlt_keep(hlt_tree)
        & _bkgcat(data_tree)
    )


def ipchi2_cut(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform IPCHI2 cut on dataframe

    """
    return dataframe[dataframe["D0 ipchi2"] < MAX_IPCHI2]
