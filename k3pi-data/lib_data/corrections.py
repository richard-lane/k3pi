"""
Functions for finding MC correction weights

"""
import pickle
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import boost_histogram as bh
from hep_ml.reweight import BinsReweighter

from . import util


def add_multiplicity_columns(tree, dataframe: pd.DataFrame, keep: np.ndarray) -> None:
    """
    Add nTracks, nLongTracks and nSPDHits columns to a dataframe in place
    Only applicable to data and MC

    :param dataframe: dataframe to add columns to
    :param keep: boolean mask of which events to keep

    """
    dataframe["n_tracks"] = tree["nTracks"].array()[keep]
    dataframe["n_long_tracks"] = tree["nLongTracks"].array()[keep]
    dataframe["n_spd_hits"] = tree["nSPDHits"].array()[keep]


def event_multiplicity(
    n_tracks_mc: np.ndarray, n_tracks_data: np.ndarray
) -> BinsReweighter:
    """
    Correct for event multiplicity by reweighting in nTracks

    Returns trained reweighter

    """
    # Want approx this number of evts in each bin
    n_bins = min(len(n_tracks_mc), len(n_tracks_data)) // 20

    # If we have more bins we want more neighbours
    weighter = BinsReweighter(n_bins=n_bins, n_neighs=n_bins // 60)

    weighter.fit(target=n_tracks_data, original=n_tracks_mc)
    return weighter


def _bin_values(
    histogram: bh.Histogram, momentum: np.ndarray, eta: np.ndarray
) -> np.ndarray:
    """
    Get histogram values from the right bins

    """
    # I don't think boost_histogram supports array indexing
    # Find indices, etc using numpy
    p_bins = histogram.axes[0].edges
    eta_bins = histogram.axes[1].edges

    p_indices = np.digitize(momentum, p_bins) - 1
    eta_indices = np.digitize(eta, eta_bins) - 1

    # Set over/underflow values to 0
    out_of_range = np.logical_or.reduce(
        (
            p_indices < 0,
            p_indices == len(p_bins) - 1,
            eta_indices < 0,
            eta_indices == len(eta_bins) - 1,
        )
    )
    retval = np.zeros_like(momentum)
    retval[~out_of_range] = histogram.values()[
        p_indices[~out_of_range], eta_indices[~out_of_range]
    ]

    return retval


def k_pi_hists(year: str, magnetisation: str) -> Tuple[bh.Histogram, bh.Histogram]:
    """
    PID calibration hists, for kaon and pion
    """
    assert year in {"2016", "2017", "2018"}
    assert magnetisation in {"magdown", "magup"}

    paths = (
        f"pidcalib_output/effhists-Turbo{year[-2:]}-{magnetisation[3:]}-{particle}-DLLK{condition}-P.ETA.pkl"
        for particle, condition in zip(("K", "Pi"), (">8.0", "<0.0"))
    )

    hists = []
    for path in paths:
        with open(path, "rb") as hist_f:
            hists.append(pickle.load(hist_f))

    return tuple(hists)


def pid_weights(
    etas: np.ndarray,
    momenta: np.ndarray,
    pion_hist: bh.Histogram,
    kaon_hist: bh.Histogram,
) -> np.ndarray:
    """
    Correct for PID efficiency (?) for a MC sample

    :param etas: (4, N) shape array of (K, pi, pi, pi) eta
    :param momenta: (4, N) shape array of (K, pi, pi, pi) momenta (magnitude)

    :returns: array of weights to apply to MC

    """
    k_vals = _bin_values(kaon_hist, momenta[0], etas[0])
    pi_vals = tuple(
        _bin_values(pion_hist, momentum, eta)
        for (momentum, eta) in zip(momenta[1:], etas[1:])
    )

    return np.multiply.reduce((k_vals, *pi_vals))


def pid_wts_df(
    dataframe: pd.DataFrame,
    pion_hist: bh.Histogram,
    kaon_hist: bh.Histogram,
) -> np.ndarray:
    """
    Find PID weights for a dataframe

    :param dataframe: a dataframe

    :returns: array of weights to apply to MC

    """
    # Find eta
    particles = util.k_3pi(dataframe)

    k_eta, pi1_eta, pi2_eta, pi3_eta = (
        util.eta(*particle[0:3]) for particle in particles
    )

    # Find total momentum
    k_p, pi1_p, pi2_p, pi3_p = (
        np.sqrt(particle[0] ** 2 + particle[1] ** 2 + particle[2] ** 2)
        for particle in particles
    )

    # Stack them
    eta = np.row_stack((k_eta, pi1_eta, pi2_eta, pi3_eta))
    momenta = np.row_stack((k_p, pi1_p, pi2_p, pi3_p))

    # Find weights
    return pid_weights(eta, momenta, pion_hist, kaon_hist)


def pid_wts_dfs(
    dataframes: Iterable[pd.DataFrame],
    pion_hist: bh.Histogram,
    kaon_hist: bh.Histogram,
) -> Iterable[np.ndarray]:
    """
    Find PID weights for dataframes; returns generators

    :param dataframe: a dataframe

    :returns: generator of array of weights to apply to MC

    """
    for dataframe in dataframes:
        yield pid_wts_df(dataframe, pion_hist, kaon_hist)
