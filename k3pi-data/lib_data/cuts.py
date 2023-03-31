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
MIN_D0_MASS, MAX_D0_MASS = 1840.83, 1888.83


def d0_mass(dataframe: pd.DataFrame) -> np.ndarray:
    """
    The best fit mass of the D0

    """
    return dataframe["Dst_ReFit_D0_M"]


def dst_mass(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Best fit mass of D*

    """
    return dataframe["Dst_ReFit_M"]


def _d0_mass_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep events near the nominal D0 mass

    """
    d0_m = d0_mass(dataframe)

    return (MIN_D0_MASS < d0_m) & (d0_m < MAX_D0_MASS)


def _uppermass_mass_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep events in the upper delta M sideband and D0 mass sidebands

    """
    d0_m = d0_mass(dataframe)
    delta_m = dst_mass(dataframe) - d0_m

    d0_keep = (d0_m < MIN_D0_MASS) | (d0_m > MAX_D0_MASS)
    dst_keep = (152.0 < delta_m) & (delta_m < 157.0)

    return d0_keep & dst_keep


def _delta_m_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep events where the D* - D0 mass difference is near the nominal pi mass

    """
    min_mass, max_mass = pdfs.domain()
    delta_m = dst_mass(dataframe) - d0_mass(dataframe)

    return (min_mass < delta_m) & (delta_m < max_mass)


def _time_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Refit time < the chosen number of lifetimes

    """
    return dataframe["time"] < MAX_LIFETIMES


def _ghost_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep events with small ghost probability

    """
    return dataframe["Dst_pi_ProbNNghost"] < 0.1


def _angle_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep events with large angles between hadrons

    """
    # Start by assuming we will keep everything
    retval = np.ones(len(dataframe), dtype=bool)

    # Find 3-momenta of each species
    suffices = "PX", "PY", "PZ"

    momenta = tuple(
        np.row_stack([dataframe[f"{prefix}_{suffix}"] for suffix in suffices])
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


def _l0_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep any events where the global L0 trigger is TIS or hadron trigger is TOS

    """
    return (dataframe["D0_L0HadronDecision_TOS"] == 1) | (
        dataframe["Dst_L0Global_TIS"] == 1
    )


def _hlt_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep any events where either the 1 or 2 track HLT1 decision for the D0 is TOS

    """
    return (dataframe["D0_Hlt1TrackMVADecision_TOS"] == 1) | (
        dataframe["D0_Hlt1TwoTrackMVADecision_TOS"] == 1
    )


def _hlt_keep_pgun(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep any events where either the 1 or 2 track HLT1 decision for the D0 is TOS

    Different branch naming convention for particle gun means we need this separate function

    """
    # Particle gun D0 HLT branch is named Dplus
    # ask Nathan Jurik if you care about why this is
    return (dataframe["Dplus_Hlt1TrackMVADecision_TOS"] == 1) | (
        dataframe["Dplus_Hlt1TwoTrackMVADecision_TOS"] == 1
    )


def _bkgcat(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep signal events only by cutting on the D0 and D* background category

    """
    return (dataframe["D0_BKGCAT"] == 0) & (dataframe["Dst_BKGCAT"] == 0)


def _pid_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Keep only events with good enough daughter PIDK

    """
    # Find kaon and pion probabilities for each daughter,
    # cut on some combination of these
    retval = np.ones(len(dataframe), dtype=bool)

    # Kaon
    retval &= dataframe["D0_P0_PIDK"] > 8

    # Opposite sign pion
    retval &= dataframe["D0_P1_PIDK"] < 0
    retval &= dataframe["D0_P3_PIDK"] < 0

    return retval


def _ipchi2_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    IPchi2 keep
    """
    return dataframe["D0_IPCHI2_OWNPV"] < MAX_IPCHI2


def _trueorigvtx_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Truth matching cut enforcing that Dst true orig vtx is not 0

    For the particle gun

    """
    return dataframe["Dst_TRUEORIGINVERTEX_X"] != 0.0


def uppermass_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Which events to keep for upper mass sideband

    PID, trigger and cuts for rejecting multiple candidates

    """
    return (
        _uppermass_mass_keep(dataframe)
        & _l0_keep(dataframe)
        & _hlt_keep(dataframe)
        & _pid_keep(dataframe)
    )


def data_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Which events to keep in real data

    Same as uppermass but with extra cuts on D0 mass and delta M

    """
    return (
        _d0_mass_keep(dataframe)
        & _delta_m_keep(dataframe)
        & _l0_keep(dataframe)
        & _hlt_keep(dataframe)
        & _pid_keep(dataframe)
    )


def mc_keep(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Which events to keep for MC

    The same as data cuts, but with BKGCAT cut (truth matching) + IPCHI2

    """
    return (
        _d0_mass_keep(dataframe)
        & _delta_m_keep(dataframe)
        & _l0_keep(dataframe)
        & _hlt_keep(dataframe)
        & _pid_keep(dataframe)
        & _bkgcat(dataframe)
        & _ipchi2_keep(dataframe)
    )


def pgun_keep(data_tree, hlt_tree) -> np.ndarray:
    """
    Which events to keep for particle gun

    Particle gun is special - there's no L0 info
    (TIS doesn't make sense, TOS info isn't in the tree)
    so we just do HLT cuts instead of HLT and L0 trigger

    """
    return (
        _d0_mass_keep(dataframe)
        & _delta_m_keep(dataframe)
        & _hlt_keep_pgun(dataframe)
        & _pid_keep(dataframe)
        & _bkgcat(dataframe)
        & _ipchi2_keep(dataframe)
        & _trueorigvtx_keep(dataframe)
    )


def ipchi2_cut(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform IPCHI2 cut on dataframe

    Unlike the other fcns in this module, this one is user-facing
    i.e. operates on the pickled dataframes, not the ones ive sliced straight from the ROOT file

    """
    return dataframe[dataframe["D0 ipchi2"] < MAX_IPCHI2]
