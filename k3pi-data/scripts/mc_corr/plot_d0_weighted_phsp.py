"""
Make histograms of D0 eta and momentum after
the reweighting

"""
import sys
import pathlib
import argparse


import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))

from lib_data import get, d0_mc_corrections, util, stats
from lib_efficiency.plotting import phsp_labels
from lib_time_fit.definitions import TIME_BINS


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    # Get testing particle gun dataframe
    pgun_df = get.particle_gun(year, sign, magnetisation, show_progress=True)
    pgun_df = pgun_df[~pgun_df["train"]]

    # Get D0 MC corr weights
    weighter = d0_mc_corrections.get_pgun(year, "cf", magnetisation)
    weights = weighter.weights(d0_mc_corrections.d0_points(pgun_df))
    weights /= np.mean(weights)

    # Find phase space points
    points = np.column_stack((helicity_param(*util.k_3pi(pgun_df)), pgun_df["time"]))
    t_bins = TIME_BINS[1:-1]

    # Only keep points in the time bins
    keep = (t_bins[0] < points[:, -1]) & (points[:, -1] < t_bins[-1])
    weights = weights[keep]
    points = points[keep]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    # Hist the phsp variables
    hist_kw = {"density": True, "histtype": "step"}
    for axis, data, label in zip(axes.ravel()[:-1], points.T[:-1], phsp_labels()[:-1]):
        contents, bins, _ = axis.hist(data, bins=100, label="Unweighted", **hist_kw)
        axis.hist(data, bins=bins, weights=weights, label="Weighted", **hist_kw)
        axis.set_ylim(0, np.max(contents) * 1.1)

        axis.set_xlabel(label)

    # Hist the times in the analysis time bins, and plot error bars
    t_axis = axes.ravel()[-1]
    counts, errs = stats.counts(points[:, -1], t_bins)
    print(weights)
    print(errs)
    weighted_counts, weighted_errs = stats.counts(
        points[:, -1], t_bins, weights=weights
    )

    centres = (t_bins[1:] + t_bins[:-1]) / 2
    widths = t_bins[1:] - t_bins[:-1]

    t_axis.errorbar(centres, counts, xerr=widths / 2, yerr=errs)
    t_axis.errorbar(centres, weighted_counts, xerr=widths / 2, yerr=weighted_errs)

    t_axis.set_ylim(0, t_axis.get_ylim()[1])

    axes[0, 0].legend()
    fig.tight_layout()

    fig.suptitle("Testing Data")

    fig.tight_layout()

    path = f"d0_distributions_{sign}_phsp.png"
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
