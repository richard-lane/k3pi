"""
Plot the decay time efficiency as a ratio of MC to AmpGen times

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))

from lib_data import get, util


def main(*, year: str, magnetisation: str):
    """
    Plot the CF and DCS efficiencies derived from MC
    as a function of time

    """
    bins = np.linspace(0.0, 10.0, 100)

    mc_cf_count, _ = np.histogram(get.mc(year, "cf", magnetisation)["time"], bins=bins)
    mc_dcs_count, _ = np.histogram(
        get.mc(year, "dcs", magnetisation)["time"], bins=bins
    )

    ag_cf_count, _ = np.histogram(get.ampgen("cf")["time"], bins=bins)
    ag_dcs_count, _ = np.histogram(get.ampgen("dcs")["time"], bins=bins)

    cf_ratio, cf_err = util.ratio_err(
        mc_cf_count, ag_cf_count, np.sqrt(mc_cf_count), np.sqrt(ag_cf_count)
    )
    dcs_ratio, dcs_err = util.ratio_err(
        mc_dcs_count, ag_dcs_count, np.sqrt(mc_dcs_count), np.sqrt(ag_dcs_count)
    )

    fig, axis = plt.subplots()
    centres = (bins[1:] + bins[:-1]) / 2
    half_widths = (bins[1:] - bins[:-1]) / 2

    axis.errorbar(
        centres,
        cf_ratio,
        xerr=half_widths,
        yerr=cf_err,
        fmt="b+",
        label="CF",
        alpha=0.5,
    )
    axis.errorbar(
        centres,
        dcs_ratio,
        xerr=half_widths,
        yerr=dcs_err,
        fmt="g+",
        label="DCS",
        alpha=0.5,
    )

    axis.legend()

    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"Efficiency")
    axis.set_yticks([])

    path = str(
        pathlib.Path(__file__).resolve().parents[0]
        / f"time_efficiency_{year}_{magnetisation}.png"
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

    main(**vars(parser.parse_args()))
