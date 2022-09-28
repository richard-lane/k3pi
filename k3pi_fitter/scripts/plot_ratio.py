"""
Plot the ratio of decay times for some different kinds of data

This doesn't actually do a fit, it just plots the ratio of decay times
to check that you've created the dataframes correctly

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import get
from lib_time_fit import definitions


def _ratio(
    dcs_t: np.ndarray, cf_t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find binned ratio of decay times, its ratio, bin centres, and bin widths

    """
    bins = np.array(definitions.TIME_BINS)

    centres = (bins[:-1] + bins[1:]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    dcs_count, _ = np.histogram(dcs_t, bins)
    cf_count, _ = np.histogram(cf_t, bins)

    ratio = dcs_count / cf_count
    err = ratio * np.sqrt(1 / dcs_count + 1 / cf_count)

    return ratio, err, centres, widths


def _plot(
    axis: plt.Axes, dcs_df: pd.DataFrame, cf_df: pd.DataFrame, label: str
) -> None:
    """
    Plot ratio of decay times on an axis

    """
    ratio, err, centres, widths = _ratio(dcs_df["time"], cf_df["time"])

    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="+", label=label)


def main():
    """
    Get dataframes, plot ratios of CF/DCS on an axis

    """
    signs = "dcs", "cf"
    year, magnetisation = "2018", "magdown"

    ag_dfs = [get.ampgen(sign) for sign in signs]
    pgun_dfs = [get.particle_gun(sign, show_progress=True) for sign in signs]
    mc_dfs = [get.mc(year, sign, magnetisation) for sign in signs]

    fig, axis = plt.subplots()

    _plot(axis, *ag_dfs, "AmpGen")
    _plot(axis, *pgun_dfs, "Particle Gun")
    _plot(axis, *mc_dfs, "MC")

    axis.set_xlabel(r"t / $\tau$")
    axis.set_ylabel(r"$WS/RS$")

    axis.plot(axis.get_xlim(), [1, 1], "k--")

    fig.suptitle("Time Ratios")

    axis.legend()

    plt.show()


if __name__ == "__main__":
    main()
