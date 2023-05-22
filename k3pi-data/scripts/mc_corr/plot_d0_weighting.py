"""
Make histograms of D0 eta and momentum after
the reweighting

"""
import sys
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


def _dataframe(year: str, sign: str, magnetisation: str) -> pd.DataFrame:
    """
    Get the right dataframe - sliced

    """
    n_dfs = 5
    dataframe = pd.concat(
        list(
            islice(
                cuts.cands_cut_dfs(
                    cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation))
                ),
                5,
                5 + n_dfs,  # different dfs to those used in training
            )
        )
    )

    # Do a time cut as well for some reason
    low_t, high_t = (TIME_BINS[1], TIME_BINS[-2])
    keep = (low_t < dataframe["time"]) & (dataframe["time"] < high_t)

    return dataframe[keep]


def main(*, year: str, magnetisation: str, sign: str, data_type: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    pgun = data_type == "pgun"
    weighter = (
        d0_mc_corrections.get_pgun(year, "cf", magnetisation)
        if pgun
        else d0_mc_corrections.get_mc(year, "cf", magnetisation)
    )

    mc_df = (
        get.particle_gun(year, sign, magnetisation, show_progress=True)
        if pgun
        else get.mc(year, sign, magnetisation)
    )
    mc_df = mc_df[~mc_df["train"]]
    mc_pts = d0_mc_corrections.d0_points(mc_df)

    data_df = _dataframe(year, sign, magnetisation)
    data_pts = d0_mc_corrections.d0_points(data_df)

    weights = weighter.weights(mc_pts)

    # Bins for plotting
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    eta_bins = np.linspace(1.5, 5.5, 100)
    p_bins = np.linspace(0.0, 500000, 100)

    hist_kw = {"histtype": "step", "density": True}
    axes[0].hist(data_pts[0], bins=eta_bins, label="Data", **hist_kw)
    axes[0].hist(
        mc_pts[0],
        bins=eta_bins,
        label="particle gun" if pgun else "MC",
        alpha=0.5,
        **hist_kw,
    )
    axes[0].hist(mc_pts[0], bins=eta_bins, label="weighted", weights=weights, **hist_kw)

    axes[1].hist(data_pts[1], bins=p_bins, label="Data", **hist_kw)
    axes[1].hist(
        mc_pts[1],
        bins=p_bins,
        label="Particle Gun" if pgun else "MC",
        alpha=0.5,
        **hist_kw,
    )
    axes[1].hist(mc_pts[1], bins=p_bins, label="Weighted", weights=weights, **hist_kw)

    axes[0].set_title(r"$\eta$")
    axes[1].set_title(r"$p$")
    axes[1].legend()

    fig.tight_layout()

    path = f"{data_type}_{sign}_d0_distributions_test_weighted.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )
    parser.add_argument("sign", type=str, help="sign of particle gun to use")
    parser.add_argument(
        "data_type", type=str, help="pgun or mc", choices={"pgun", "mc"}
    )

    main(**vars(parser.parse_args()))
