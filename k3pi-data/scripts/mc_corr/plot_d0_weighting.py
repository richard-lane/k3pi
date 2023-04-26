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


def _dataframe(year: str, magnetisation: str) -> pd.DataFrame:
    """
    Get the right dataframe - sliced

    """
    # TODO change this to use a testing part of the data...
    n_dfs = 3
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
    low_t, high_t = (TIME_BINS[0], TIME_BINS[-2])
    keep = (low_t < dataframe["time"]) & (dataframe["time"] < high_t)

    return dataframe[keep]


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    weighter = d0_mc_corrections.get_pgun(year, "cf", magnetisation)

    pgun_df = get.particle_gun(year, sign, magnetisation, show_progress=True)
    pgun_df = pgun_df[~pgun_df["train"]]
    pgun_pts = d0_mc_corrections.d0_points(pgun_df)

    data_df = _dataframe(year, magnetisation)
    data_pts = d0_mc_corrections.d0_points(data_df)

    weights = weighter.weights(pgun_pts)

    # Bins for plotting
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    eta_bins = np.linspace(1.5, 5.5, 100)
    p_bins = np.linspace(0.0, 500000, 100)

    hist_kw = {"histtype": "step", "density": True}
    axes[0].hist(data_pts[0], bins=eta_bins, label="Data", **hist_kw)
    axes[0].hist(pgun_pts[0], bins=eta_bins, label="Particle Gun", alpha=0.5, **hist_kw)
    axes[0].hist(
        pgun_pts[0], bins=eta_bins, label="Particle Gun", weights=weights, **hist_kw
    )

    axes[1].hist(data_pts[1], bins=p_bins, label="Data", **hist_kw)
    axes[1].hist(pgun_pts[1], bins=p_bins, label="Particle Gun", alpha=0.5, **hist_kw)
    axes[1].hist(
        pgun_pts[1], bins=p_bins, label="Particle Gun", weights=weights, **hist_kw
    )

    axes[0].set_title(r"$\eta$")
    axes[1].set_title(r"$p$")
    axes[1].legend()

    fig.suptitle("Training Data")

    fig.tight_layout()

    path = "d0_distributions_test_weighted.png"
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

    main(**vars(parser.parse_args()))
