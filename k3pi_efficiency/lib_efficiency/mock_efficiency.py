"""
Toy efficiency for applying to AmpGen, doing studies, etc.

"""
import numpy as np
import pandas as pd

from . import time_fitter
from .efficiency_definitions import MIN_TIME


def time_efficiency(dataframe: pd.DataFrame, factor: float = 1.0) -> np.ndarray:
    """
    Use the time efficiency that we fit for the reweighter

    :param dataframe: data to find efficiency for
    :param factor: a numerical factor that one could use to
                   modify the efficiency

    :returns: array of efficiencies, between 0 and 1
    :raises AssertionError: if the efficiencies exceed 1 somehow

    """
    # Straight line
    scale = 1.0
    return factor * dataframe["time"] / scale - MIN_TIME * factor / scale

    times = dataframe["time"]
    retvals = (
        time_fitter.normalised_pdf(times, MIN_TIME, 1.0, 2.0, 1.0, 2.0, 1.0)[1]
        * np.exp(factor * times)
        / 10
    )

    assert np.max(retvals) < 1.0
    assert np.min(retvals) >= 0.0

    return retvals


def phsp_efficiency(dataframe: pd.DataFrame, factor: float = 1.0) -> np.ndarray:
    """
    Phase space efficiency is just something from the pT of the Kaon

    :param dataframe: data to find efficiency for
    :param factor: a numerical factor that one could use to
                   modify the efficiency

    :returns: array of efficiencies, between 0 and 1
    :raises AssertionError: if the efficiencies exceed 1 somehow

    """
    # Find the transverse momentum of the kaon
    k_pt = np.sqrt(dataframe["Kplus_Px"] ** 2 + dataframe["Kplus_Py"] ** 2)

    # Efficiency is this
    efficiency = 1.0 - 0.25 * factor * np.sin(np.pi * k_pt / 1000)

    assert np.max(efficiency < 1.0)
    assert np.min(efficiency > 0.0)

    return efficiency


def combined_efficiency(
    dataframe: pd.DataFrame, time_factor: float = 1.0, phsp_factor: float = 1.0
) -> np.ndarray:
    """
    Combined phsp and time efficiencies

    """
    return time_efficiency(dataframe, time_factor) * phsp_efficiency(
        dataframe, phsp_factor
    )


def _keep(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    """
    Events to keep

    """
    rnd = rng.random(len(values))
    return rnd < values


def accepted(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
    *,
    time_factor: float,
    phsp_factor: float
) -> np.ndarray:
    """
    Mask of whether events get accepted by the efficiency

    Prints overall efficiency too

    """
    time_effs = _keep(rng, time_efficiency(dataframe, time_factor))
    phsp_effs = _keep(rng, phsp_efficiency(dataframe, phsp_factor))

    # Keep mask is whether the random number is < the efficiency at that point
    return time_effs & phsp_effs
