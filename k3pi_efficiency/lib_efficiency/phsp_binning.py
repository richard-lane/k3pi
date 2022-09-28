"""
Functions for phase space binning + numerically evaluating the coherence factor

"""
from typing import Tuple
import numpy as np

from .amplitude_models import amplitudes
from .amplitude_models import definitions


def coherence_factor(
    k: np.ndarray,
    pi1: np.ndarray,
    pi2: np.ndarray,
    pi3: np.ndarray,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure coherence factor using a numerical integration
    The amplitudes are evaluated for the decay K+pi-pi-pi+; if you are
    dealing with the conjugate decay you should multiply your 3-momenta
    by -1.

    :param k: (4, N) shape numpy array of k parameters (px, py, pz, E)
    :param pi1: (4, N) shape numpy array of pi1 parameters (px, py, pz, E)
    :param pi2: (4, N) shape numpy array of pi2 parameters (px, py, pz, E)
    :param pi3: (4, N) shape numpy array of pi3 parameters (px, py, pz, E)
    :param weights: weighting to apply to events

    :returns: numpy array of coherence factor magnitudes
    :returns: numpy array of coherence factor phases, in degrees

    """
    if weights is None:
        weights = np.ones(len(k.T))

    # Find amplitudes
    # We're assuming K+
    cf = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, +1)
    dcs = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, +1) * definitions.DCS_OFFSET

    # Find Z and integrals
    z = np.sum(cf * dcs.conjugate() * weights)
    num_dcs = np.sum((np.abs(dcs) ** 2) * weights)
    num_cf = np.sum((np.abs(cf) ** 2) * weights)

    # Find R, d and return
    R = abs(z) / np.sqrt(num_dcs * num_cf)
    d = np.angle(z, deg=True)

    return R, d
