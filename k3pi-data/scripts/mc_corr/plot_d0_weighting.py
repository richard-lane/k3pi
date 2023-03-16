"""
Make histograms of D0 eta and momentum after
the reweighting

"""
import sys
import pathlib
import argparse
from typing import Tuple
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import get, d0_mc_corrections, ipchi2_fit


def _dataframe(year: str, magnetisation: str) -> pd.DataFrame:
    """
    Get the right dataframe - sliced

    """
    n_dfs = 1
    retval = pd.concat(list(islice(get.data(year, "cf", magnetisation), n_dfs)))

    low_t, high_t = (0.0, 19.0)
    keep = (low_t < retval["time"]) & (retval["time"] < high_t)

    return retval[keep]


def _ipchi2s(dataframe: pd.DataFrame) -> np.ndarray:
    """
    ipchi2

    """
    return np.log(dataframe["D0 ipchi2"])


def _ip_fit(ipchi2s: np.ndarray) -> Tuple[Tuple, np.ndarray]:
    """
    Returns fit params + ipchi2s

    """
    sig_defaults = {
        "centre_sig": 0.9,
        "width_l_sig": 1.6,
        "width_r_sig": 0.9,
        "alpha_l_sig": 0.0,
        "alpha_r_sig": 0.0,
        "beta_sig": 0.0,
    }
    bkg_defaults = {
        "centre_bkg": 4.5,
        "width_bkg": 1.8,
        "alpha_bkg": 0.0,
        "beta_bkg": 0.0,
    }
    fitter = ipchi2_fit.unbinned_fit(ipchi2s, 0.6, sig_defaults, bkg_defaults)

    return fitter.values, ipchi2s


def _plot_d0_points(
    axes: Tuple[plt.Axes, plt.Axes], d0_pts: np.ndarray, sweights: np.ndarray
) -> None:
    """
    Plot the D0 eta and P on two axes, weighted and not

    """
    d0_eta, d0_p = d0_pts

    hist_kw = {"histtype": "step"}
    n_bins = 100
    eta_bins = np.linspace(1.8, 5.5, n_bins)
    p_bins = np.linspace(0.0, 550_000, n_bins)

    axes[0].hist(d0_eta, **hist_kw, bins=eta_bins, color="k", label="Raw")
    axes[1].hist(d0_p, **hist_kw, bins=p_bins, color="k", label="Raw")

    axes[0].hist(
        d0_eta, **hist_kw, bins=eta_bins, color="r", weights=sweights, label="sWeighted"
    )
    axes[1].hist(
        d0_p, **hist_kw, bins=p_bins, color="r", weights=sweights, label="sWeighted"
    )

    axes[0].set_xlabel(r"$D^0 \eta$")
    axes[1].set_xlabel(r"$D^0 P$")

    axes[0].legend()


def main(*, year: str, magnetisation: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    weighter = d0_mc_corrections.get_pgun(year, "cf", magnetisation)

    # Bins for plotting
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))

    weighter.plot_distribution("target", bins, "d0_correction_data.png")
    weighter.plot_distribution("original", bins, "d0_correction_pgun.png")
    weighter.plot_ratio(bins, "d0_correction_data_pgun_ratio.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )

    main(**vars(parser.parse_args()))
