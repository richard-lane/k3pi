"""
Utilities and stuff for phase space binning

"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from . import util
from lib_efficiency.amplitude_models import amplitudes, definitions as ampdef

KS_MASS_MEV = 497.611
KS_VETO_WIDTH_MEV = 10.0

BINS = (-180.0, -39.0, 0.0, 43.0, 180.0)


def _veto(pi_minus: np.ndarray, pi_plus: np.ndarray) -> np.ndarray:
    """
    Mask of events that are veto'd

    """
    mass = util.invariant_masses(*np.add(pi_minus, pi_plus))
    return np.abs(mass - KS_MASS_MEV) < KS_VETO_WIDTH_MEV


def ks_veto(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Mask of events veto'd since they are too close to the
    nominal Ks mass

    :param dataframe: events
    :returns: bool mask of events that are veto'd

    """
    _, pi1_m, pi2_m, pi3_p = tuple(a.astype(np.float64) for a in util.k_3pi(dataframe))

    veto = np.full(len(dataframe), False)
    veto |= _veto(pi1_m, pi3_p)
    veto |= _veto(pi2_m, pi3_p)

    return veto


def interference_terms(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Amplitude interference cross term for a dataframe

    Dataframe must contain a K ID column

    TODO maybe refactor to also accept a mask of K+ type
    events if we need to also apply this binning to
    MC or AmpGen

    :param dataframe: events
    :returns: array of complex cross terms

    """
    k_ids = dataframe["K ID"]
    assert (np.unique(k_ids) == np.array([-321, 321])).all(), "check only K (321)"

    # Create a mask of K+, K- arrays
    k_plus = k_ids == 321

    k3pi = tuple(a.astype(np.float64) for a in util.k_3pi(dataframe))
    retval = np.empty(len(dataframe), dtype=complex)

    # Find the K- amplitudes
    k3pi_plus = [p[:, k_plus] for p in k3pi]
    plus_cf = amplitudes.cf_amplitudes(*k3pi_plus, +1)
    plus_dcs = amplitudes.dcs_amplitudes(*k3pi_plus, +1) * ampdef.DCS_OFFSET

    # Find the K- amplitudes
    k3pi_minus = [p[:, ~k_plus] for p in k3pi]
    minus_cf = amplitudes.cf_amplitudes(*k3pi_minus, -1)
    minus_dcs = amplitudes.dcs_amplitudes(*k3pi_minus, -1) * ampdef.DCS_OFFSET

    # Find the angle between them
    retval[k_plus] = plus_dcs.conj() * plus_cf
    retval[~k_plus] = minus_dcs.conj() * minus_cf

    return retval
