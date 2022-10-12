"""
Functions for finding MC correction weights

"""
import numpy as np
import pandas as pd
import boost_histogram as bh
from hep_ml.reweight import BinsReweighter


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
