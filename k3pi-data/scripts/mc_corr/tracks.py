"""
Show MC correction reweighting in the number of tracks + related vars

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import get, corrections, plotting


def _track_hists(
    axes: Tuple[plt.Axes, plt.Axes],
    bins: np.ndarray,
    mc_tracks: np.ndarray,
    data_tracks: np.ndarray,
    weights: np.ndarray,
) -> None:
    """
    Plot hist and pulls

    """
    hist_ax, pull_ax = axes
    hist_kw = {"histtype": "step", "bins": bins}
    hist_ax.hist(mc_tracks, label="MC", color="r", **hist_kw)
    hist_ax.hist(data_tracks, label="Data", color="g", **hist_kw)
    hist_ax.hist(
        mc_tracks, label="MC (Reweighted)", color="b", **hist_kw, weights=weights
    )

    # plotting.pull(
    #     pull_ax,
    #     bins,
    #     (mc_tracks, data_tracks),
    #     (None, None),
    #     fmt="k.",
    #     markersize=0.5,
    #     label="Before",
    # )
    # plotting.pull(
    #     pull_ax,
    #     bins,
    #     (mc_tracks, data_tracks),
    #     (weights, None),
    #     fmt="r.",
    #     markersize=0.5,
    #     label="After",
    # )


def _ntrack_hists(mc_tracks, mc_train, data_tracks, data_train, train_wt, test_wt):
    """
    plot test/train hists

    """
    # Plot train and test histograms
    fig, ax = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(10, 5)
    )

    bins = np.linspace(-1, 155, 20)
    _track_hists(
        (ax["A"], ax["C"]), bins, mc_tracks[mc_train], data_tracks[data_train], train_wt
    )
    _track_hists(
        (ax["B"], ax["D"]),
        bins,
        mc_tracks[~mc_train],
        data_tracks[~data_train],
        test_wt,
    )

    ax["A"].legend()
    ax["C"].legend()

    ax["A"].set_title("Train")
    ax["B"].set_title("Test")
    fig.tight_layout()

    fig.savefig("track_multiplicity.png")


def _long_track_hists(
    mc_long_tracks: np.ndarray, data_long_tracks: np.ndarray, weights: np.ndarray
) -> None:
    """
    Plot histograms of nLongTracks

    """
    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(8, 8))

    bins = np.linspace(-1, 150, 20)
    _track_hists((ax["A"], ax["B"]), bins, mc_long_tracks, data_long_tracks, weights)

    ax["A"].legend()
    ax["B"].legend()

    ax["A"].set_title("Long Tracks")
    fig.tight_layout()

    fig.savefig("long_tracks.png")


def _spd_hists(
    mc_spd_hits: np.ndarray, data_spd_hits: np.ndarray, weights: np.ndarray
) -> None:
    """
    Plot histograms of nSPDHits

    """
    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(8, 8))

    bins = np.linspace(-1, 1200, 50)
    _track_hists((ax["A"], ax["B"]), bins, mc_spd_hits, data_spd_hits, weights)

    ax["A"].legend()
    ax["B"].legend()

    ax["A"].set_title("SPD Hits")
    fig.tight_layout()

    fig.savefig("spd_hits.png")


def _track_multiplicity(
    data_df: pd.DataFrame, mc_df: pd.DataFrame, data_train: np.ndarray
):
    """
    Track multiplicity histograms

    """
    # Reweight MC to data
    mc_train = mc_df["train"]
    mc_tracks = mc_df["n_tracks"]
    data_tracks = data_df["n_tracks"]

    # Don't care about stuff with loads of tracks cus its not very common
    max_tracks = 150
    mc_keep = mc_tracks < max_tracks
    data_keep = data_tracks < max_tracks

    mc_train = mc_train[mc_keep]
    data_train = data_train[data_keep]
    mc_tracks = mc_tracks[mc_keep]
    data_tracks = data_tracks[data_keep]

    reweighter = corrections.event_multiplicity(
        mc_tracks[mc_train], data_tracks[data_train]
    )
    train_wt = reweighter.predict_weights(mc_tracks[mc_train])
    test_wt = reweighter.predict_weights(mc_tracks[~mc_train])

    _ntrack_hists(mc_tracks, mc_train, data_tracks, data_train, train_wt, test_wt)

    # Use these weights to plot histograms of nLongTracks
    _long_track_hists(
        mc_df["n_long_tracks"][mc_keep][~mc_train],
        data_df["n_long_tracks"][data_keep][~data_train],
        test_wt,
    )

    # And nSPDHits
    _spd_hists(
        mc_df["n_spd_hits"][mc_keep][~mc_train],
        data_df["n_spd_hits"][data_keep][~data_train],
        test_wt,
    )


def main(args):
    """
    Plot variables used in the MC correction reweighting

    """
    # Read dataframes
    data_df = pd.concat(
        get.data(args.year, args.sign, args.magnetisation), ignore_index=True
    )
    mc_df = get.mc(args.year, args.sign, args.magnetisation)

    # Split data into test/train
    data_train = np.random.random(len(data_df)) < 0.5

    _track_multiplicity(data_df, mc_df, data_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show effect of MC corrections")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )

    main(parser.parse_args())
