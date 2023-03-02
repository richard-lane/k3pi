"""
Plot the decay time reweighting

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import get, stats
from lib_efficiency import time_fitter
from lib_efficiency.reweighter import TimeWeighter
from lib_efficiency.efficiency_definitions import MIN_TIME


def main(*, year: str, magnetisation: str, sign: str, fit: bool):
    """
    Train a time reweighter, make a plot of the reweighting using
    an adaptive histogram

    """
    ag_df = get.ampgen(sign)
    pgun_df = get.particle_gun(sign)

    ag_time = ag_df["time"]
    pgun_time = pgun_df["time"]
    ag_train = ag_df["train"]
    pgun_train = pgun_df["train"]

    # Train a reweighter
    reweighter = TimeWeighter(min_t=MIN_TIME, fit=fit, n_bins=40000, n_neighs=10)
    reweighter.fit(mc_times=pgun_time[pgun_train], ampgen_times=ag_time[ag_train])

    # Find test weights
    wts = reweighter.correct_efficiency(pgun_time[~pgun_train])

    # Will need to scale weights up if we did a fit
    if fit:
        n_ag_above_min = np.sum(ag_time[~ag_train] > MIN_TIME)
        wts *= n_ag_above_min / np.sum(wts)

    # Plot
    fig, axis = plt.subplots()
    hist_kw = {"bins": np.linspace(MIN_TIME, 10.0, 100)}

    axis.hist(
        pgun_time[~pgun_train],
        **hist_kw,
        color="r",
        histtype="stepfilled",
        label="Particle Gun Weighted",
        alpha=0.3,
        weights=wts,
    )
    axis.hist(ag_time[~ag_train], **hist_kw, color="k", histtype="step", label="Target")
    axis.hist(
        pgun_time[~pgun_train],
        **hist_kw,
        color="r",
        histtype="step",
        label="Particle Gun MC",
    )

    # If we did a fit, also plot this
    if fit:

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return time_fitter.normalised_pdf(x, *reweighter.fitter.fit_vals)[1]

        # Scale up so its normalised in the right region
        bin_width = hist_kw["bins"][1] - hist_kw["bins"][0]
        integral_pts = np.linspace(MIN_TIME, hist_kw["bins"][-1], 1000)
        scale_factor = (
            bin_width
            * np.sum(pgun_time[~pgun_train] > MIN_TIME)
            / stats.integral(integral_pts, fitted_pdf(integral_pts))
        )

        pts = np.linspace(0.0, hist_kw["bins"][-1], 1000)
        axis.plot(pts, scale_factor * fitted_pdf(pts), "r--")

    # Reverse order of legend labels
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1])

    axis.set_xlabel(r"t/$\tau$")

    fig.tight_layout()

    path = str(
        pathlib.Path(__file__).resolve().parents[0]
        / f"time_reweight_{year}_{magnetisation}_{sign}_{fit=}.png"
    )
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot stacked hists showing the phase space binning and parameterisation"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="magnetisation direction",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="whether to fit",
    )

    main(**vars(parser.parse_args()))
