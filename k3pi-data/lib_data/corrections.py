"""
Functions for finding MC correction weights

"""
import numpy as np
import pandas as pd
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
