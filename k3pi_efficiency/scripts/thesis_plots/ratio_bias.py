"""
Show the decay time bias we see in MC

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import get
from lib_efficiency import plotting


def main(*, year: str, magnetisation: str):
    """
    Get the dataframes, plot the time ratio, show the bias

    """
    dcs_ag = get.ampgen("dcs")["time"]
    cf_ag = get.ampgen("cf")["time"]

    dcs_mc = get.mc(year, "dcs", magnetisation)["time"]
    cf_mc = get.mc(year, "cf", magnetisation)["time"]

    time_bins = np.concatenate(([0.0], np.linspace(1.0, 6.0, 15), [8.0]))

    # Cut so we only get central times
    dcs_ag = dcs_ag[(time_bins[0] < dcs_ag) & (dcs_ag < time_bins[-1])]
    cf_ag = cf_ag[(time_bins[0] < cf_ag) & (cf_ag < time_bins[-1])]

    dcs_mc = dcs_mc[(time_bins[0] < dcs_mc) & (dcs_mc < time_bins[-1])]
    cf_mc = cf_mc[(time_bins[0] < cf_mc) & (cf_mc < time_bins[-1])]

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD",
        figsize=(10, 5),
        sharex=True,
    )

    # Plot ratios
    plotting._plot_ratio(axes["A"], dcs_ag, cf_ag, None, None, time_bins)
    plotting._plot_ratio(axes["B"], dcs_mc, cf_mc, None, None, time_bins)

    # Plot hists
    plot_bins = np.linspace(time_bins[0], time_bins[-1], 100)
    hist_kw = {"x": plot_bins[:-1], "bins": plot_bins}
    dcs_kw = {**hist_kw, "color": "r", "label": "DCS", "alpha": 0.7}
    cf_kw = {**hist_kw, "color": "k", "label": "CF", "histtype": "step"}

    bin_widths = plot_bins[1:] - plot_bins[:-1]
    dcs_ag_count = np.histogram(dcs_ag, bins=plot_bins)[0] / bin_widths
    cf_ag_count = np.histogram(cf_ag, bins=plot_bins)[0] / bin_widths

    dcs_mc_count = np.histogram(dcs_mc, bins=plot_bins)[0] / bin_widths
    cf_mc_count = np.histogram(cf_mc, bins=plot_bins)[0] / bin_widths

    axes["A"].legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color="k",
                marker="+",
                linestyle=None,
                label=r"$\frac{DCS}{CF}$ Ratio",
            )
        ]
    )

    axes["C"].hist(**dcs_kw, weights=dcs_ag_count)
    axes["C"].hist(**cf_kw, weights=cf_ag_count)

    axes["D"].hist(**dcs_kw, weights=dcs_mc_count)
    axes["D"].hist(**cf_kw, weights=cf_mc_count)

    axes["C"].legend()

    for axis in axes.values():
        axis.set_xlim(time_bins[0], time_bins[-1])

    for axis in (axes["C"], axes["D"]):
        axis.set_xlabel(r"time / $\tau$")

    axes["A"].text(0.5, 1.05, "AmpGen")
    axes["B"].text(0.5, 1.05, "LHCb MC")

    fig.tight_layout()

    fig.savefig("show_ratio_bias.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the decay time ratio bias in LHCb MC"
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

    main(**vars(parser.parse_args()))
