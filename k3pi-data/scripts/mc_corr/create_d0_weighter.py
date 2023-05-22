"""
Create a weighted for correcting D0 momentum
distributions

"""
import sys
import pickle
import pathlib
import argparse
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))

from lib_data import get, d0_mc_corrections, cuts
from lib_time_fit.definitions import TIME_BINS


def _dataframe(year: str, magnetisation: str) -> pd.DataFrame:
    """
    Get the right dataframe - sliced

    """
    n_dfs = 5
    dataframe = pd.concat(
        list(
            islice(
                cuts.cands_cut_dfs(
                    cuts.ipchi2_cut_dfs(get.data(year, "cf", magnetisation))
                ),
                n_dfs,
            )
        )
    )

    # Do a time cut as well for some reason
    low_t, high_t = (TIME_BINS[1], TIME_BINS[-2])
    keep = (low_t < dataframe["time"]) & (dataframe["time"] < high_t)

    return dataframe[keep]


def _train_proj(
    weighter: d0_mc_corrections.EtaPWeighter,
    data_pts: np.ndarray,
    mc_pts: np.ndarray,
    path: str,
) -> None:
    """
    Plot projections of the training data after reweighting

    """
    weights = weighter.weights(mc_pts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    eta_bins = np.linspace(1.5, 5.5, 100)
    p_bins = np.linspace(0.0, 500000, 100)

    hist_kw = {"histtype": "step", "density": True}
    axes[0].hist(data_pts[0], bins=eta_bins, label="Data", **hist_kw)
    axes[0].hist(mc_pts[0], bins=eta_bins, label="MC", alpha=0.5, **hist_kw)
    axes[0].hist(mc_pts[0], bins=eta_bins, label="MC", weights=weights, **hist_kw)

    axes[1].hist(data_pts[1], bins=p_bins, label="Data", **hist_kw)
    axes[1].hist(mc_pts[1], bins=p_bins, label="MC", alpha=0.5, **hist_kw)
    axes[1].hist(mc_pts[1], bins=p_bins, label="MC", weights=weights, **hist_kw)

    axes[0].set_title(r"$\eta$")
    axes[1].set_title(r"$p$")
    axes[1].legend()

    fig.suptitle("Training Data")

    fig.tight_layout()

    print(f"plotting {path}")
    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)

    fig.savefig(path)


def main(*, year: str, magnetisation: str, data_type: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    sign = "cf"

    pgun = data_type == "pgun"
    # Get the D0 eta and P points
    mc_df = (
        get.particle_gun(year, sign, magnetisation, show_progress=True)
        if pgun
        else get.mc(year, sign, magnetisation)
    )
    mc_df = mc_df[mc_df["train"]]
    mc_pts = d0_mc_corrections.d0_points(mc_df)

    data_df = _dataframe(year, magnetisation)
    data_pts = d0_mc_corrections.d0_points(data_df)

    # Create reweighter
    weighter = d0_mc_corrections.EtaPWeighter(
        data_pts,
        mc_pts,
        n_bins=75,
        n_neighs=0.5,
    )

    # Plot the data used for training
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))
    weighter.plot_distribution(
        "target", bins, f"{data_type}_d0_distributions_{sign}_target.png"
    )
    weighter.plot_distribution(
        "original", bins, f"{data_type}_d0_distributions_{sign}_original.png"
    )
    weighter.plot_ratio(bins, f"{data_type}_d0_distributions_{sign}_ratio.png")

    # Plot the projections after training
    _train_proj(
        weighter,
        data_pts,
        mc_pts,
        f"{data_type}_d0_distributions_{sign}_train_weighted.png",
    )

    # Store the reweighter in a pickle dump
    d0_mc_corrections.weighter_dir().mkdir(exist_ok=True)

    dump_path = (
        d0_mc_corrections.pgun_path(year, sign, magnetisation)
        if pgun
        else d0_mc_corrections.mc_path(year, sign, magnetisation)
    )

    with open(str(dump_path), "wb") as weighter_f:
        pickle.dump(weighter, weighter_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P + a reweighter"
    )
    parser.add_argument(
        "year", type=str, help="data taking year", choices={"2018", "2017"}
    )
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )
    parser.add_argument(
        "data_type", type=str, help="pgun or mc", choices={"pgun", "mc"}
    )

    main(**vars(parser.parse_args()))
