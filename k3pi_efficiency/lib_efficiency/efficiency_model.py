"""
User interface for the efficiency reweighting

"""

import sys
import pickle
import pathlib
import numpy as np
from fourbody.param import helicity_param

from . import efficiency_definitions
from .reweighter import EfficiencyWeighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import util


def weights(
    k: np.ndarray,
    pi1: np.ndarray,
    pi2: np.ndarray,
    pi3: np.ndarray,
    t: np.ndarray,
    k_sign: str,
    year: str,
    sign: str,
    magnetisation: str,
    fit: bool,
    cut: bool,
    verbose=False,
) -> np.ndarray:
    """
    Return an estimate of weights needed to correct for detector efficiency for a series
    of D->K pi1 pi2 pi3 events.

    :param k: 2d numpy array of K data (k_px, k_py, k_pz, k_e) in GeV. Shape (4, N).
    :param pi1: 2d numpy array of pi1 data (pi1_px, pi1_py, pi1_pz, pi1_e) in GeV. Shape (4, N).
                This pion has opposite charge to the kaon.
    :param pi2: 2d numpy array of pi2 data (pi2_px, pi2_py, pi2_pz, pi2_e) in GeV. Shape (4, N).
                This pion has opposite charge to the kaon.
    :param pi3: 2d numpy array of pi3 data (pi3_px, pi3_py, pi3_pz, pi3_e) in GeV. Shape (4, N).
                This pion has the same charge as the kaon.
    :param t: 1d numpy arrays of decay times in lifetimes.
    :param k_sign: "k_plus", "k_minus" or "both"
    :param k_id: particle id of the kaon: -321 for K-, 321 for K+. This is used to flip the sign
                 of the particles' 3 momenta.
    :param year: data taking year.
    :param sign: either "RS" or "WS"
    :param magnetisation: either "MagUp" or "MagDown"
    :param fit: whether to use the reweighter trained using a fit to decay times (fit=True) or a
                histogram division (fit=False)
    :param cut: whether to use a reweighter trained on data after the BDT cut was applied
    :param verbose: whether to print a small amount of extra information

    :returns: length-N array of weights

    """
    assert efficiency_definitions.reweighter_exists(
        year, sign, magnetisation, k_sign, fit, cut
    )

    if verbose:
        print(
            f"Finding {sign} efficiencies for\n\tYear:\t{int(year)}\n\tMag:\t{magnetisation}"
            f"\n\t{k_sign=}\n\tN:\t{len(k.T)}"
        )
        print(
            f"{np.sum(t < efficiency_definitions.MIN_TIME)} times below minimum"
            f"({efficiency_definitions.MIN_TIME})"
        )

    # Find the right reweighter to unpickle
    reweighter_path = efficiency_definitions.reweighter_path(
        year,
        sign,
        magnetisation,
        k_sign,
        fit,
        cut,
    )

    # Open the reweighter
    if verbose:
        print(f"Opening reweighter at {reweighter_path}")
    with open(reweighter_path, "rb") as f:
        reweighter: EfficiencyWeighter = pickle.load(f)

    # Momentum order
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    # Parameterise event into 5+1d space
    parameterised_evts = np.column_stack(
        (
            helicity_param(k, pi1, pi2, pi3),
            t,
        )
    )

    retval = reweighter.weights(parameterised_evts)
    if verbose:
        print(f"{np.sum(retval == 0.0)} weights exactly 0.0")

    # Typically we only expect to get a weight of exactly 0 if our points are outside of the time
    # bins provided. This should only really happen if points are below the minimum time
    # This usually means that you've changed efficiency_definitions.MIN_TIME since the
    # reweighter was trained
    assert np.sum(retval == 0.0) == np.sum(t < efficiency_definitions.MIN_TIME)

    return retval
