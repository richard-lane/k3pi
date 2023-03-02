"""
Make histograms of D0 eta and momentum;
show them and save to a dump somewhere

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, d0_mc_corrections


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get particle gun, MC and data dataframes
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    pgun_pts = d0_mc_corrections.d0_points(get.particle_gun(sign))
    mc_pts = d0_mc_corrections.d0_points(get.mc(year, sign, magnetisation))
    data_pts = d0_mc_corrections.d0_points(get.data(year, sign, magnetisation))

    # Find sWeights

    # Plot mass fit from sWeighting, the sWeighted RS distributions
    # and the RS/WS distributions (before sWeighting?) to show that they're
    # the same

    # Bins for plotting
    # The reweighter doesn't use these bins, it finds its own
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))

    # Weight pgun -> data as a test
    weighter = d0_mc_corrections.EtaPWeighter(
        data_pts,
        pgun_pts,
        n_bins=100,
        n_neighs=0.5,
    )

    weighter.plot_distribution("target", bins, "d0_correction_data.png")
    weighter.plot_distribution("original", bins, "d0_correction_pgun.png")
    weighter.plot_ratio(bins, "d0_correction_data_pgun_ratio.png")

    # Make plots showing the reweighting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_kw = {"histtype": "step", "density": True}
    axes[0].hist(mc_pts[0], bins=bins[0], label="mc", **plot_kw)
    axes[0].hist(data_pts[0], bins=bins[0], label="data", **plot_kw)
    axes[0].hist(
        mc_pts[0],
        bins=bins[0],
        label="weighted",
        weights=weighter.weights(mc_pts),
        **plot_kw,
    )

    axes[1].hist(mc_pts[1], bins=bins[1], label="mc", **plot_kw)
    counts, _, _ = axes[1].hist(data_pts[1], bins=bins[1], label="data", **plot_kw)
    axes[1].hist(
        mc_pts[1],
        bins=bins[1],
        label="weighted",
        weights=weighter.weights(mc_pts),
        **plot_kw,
    )
    axes[1].set_ylim(0.0, 1.2 * np.max(counts))
    axes[0].legend()

    fig.savefig("d0_correction_data_to_pgun.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )
    parser.add_argument("sign", type=str, help="decay type", choices={"dcs", "cf"})

    main(**vars(parser.parse_args()))
