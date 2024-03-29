"""
Make histograms of D0 eta and momentum;
show them and save to a dump somewhere

"""
import sys
import pickle
import pathlib
import argparse
from typing import Tuple, Callable, Iterable
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_mass_fit"))

from lib_data import get, d0_mc_corrections, ipchi2_fit
from libFit import util as mass_util, pdfs


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


def _sweight_fcn(ipchi2: np.ndarray) -> Tuple[Callable, plt.Figure, Iterable[plt.Axes]]:
    """
    Find a unary function that projects signal out, given
    delta M

    Uses CF for the sWeighting

    Also plots information relating to the mass fit and returns the figure and axes

    """
    # Do the ip fit, get parameters and array of ipchi2s used for the fit
    fit_values, ipchi2s = _ip_fit(ipchi2)

    # Plot the fit on Axes
    fig, axes = plt.subplot_mosaic(
        "AAAACC\nAAAACC\nAAAADD\nAAAADD\nAAAAEE\nBBBBEE", figsize=(15, 12)
    )
    ip_bins = np.linspace(*ipchi2_fit.domain(), 100)
    counts, _ = np.histogram(ipchi2s, ip_bins)
    ipchi2_fit.plot(
        (axes["A"], axes["B"]),
        ip_bins,
        counts,
        np.sqrt(counts),
        fit_values,
    )

    return ipchi2_fit.sweight_fcn(fit_values), fig, axes


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


def main(*, year: str, magnetisation: str, mc_type: str):
    """
    Get particle gun, MC and data dataframe
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    # Real data dataframe
    cf_df = _dataframe(year, magnetisation)
    print(f"{len(cf_df)=}")

    # Do + plot IP fit, find sWeighting fcn
    ipchi2 = _ipchi2s(cf_df)
    sweight_fcn, fig, axes = _sweight_fcn(ipchi2)

    # get sWeights for the CF dataframe
    # training data - but probably fine?
    sweights = sweight_fcn(ipchi2)

    # Get histogram of delta M counts, plot it on one of the axes
    delta_m = mass_util.delta_m(cf_df)
    hist_kw = {
        "x": delta_m,
        "bins": np.linspace(*pdfs.domain(), 250),
        "histtype": "step",
    }
    axes["E"].hist(**hist_kw, color="k")
    axes["E"].hist(**hist_kw, color="r", weights=sweights)
    axes["E"].set_xlabel(r"$\Delta M$ / MeV")

    # Get the D0 eta and P points
    data_pts = d0_mc_corrections.d0_points(cf_df)

    # Plot these, sweighted/otherwise on the figure
    _plot_d0_points((axes["C"], axes["D"]), data_pts, sweights)

    # Save the figure
    fig.suptitle(f"RS {year} {magnetisation} sWeight IP$\\chi^2$ fit")
    fig.tight_layout()
    path = f"d0_mc_corr_{year}_{magnetisation}_to_{mc_type}_sweight_fit.png"
    print(f"plotting {path}")
    fig.savefig(path)
    plt.close(fig)

    # Weight pgun -> data as a test
    mc_dfs = (
        get.particle_gun("cf")
        if mc_type == "pgun"
        else get.mc(year, "cf", magnetisation)
    )
    pgun_pts = d0_mc_corrections.d0_points(mc_dfs)

    weighter = d0_mc_corrections.EtaPWeighter(
        data_pts,
        pgun_pts,
        target_wt=sweights,
        n_bins=200,
        n_neighs=2.0,
    )

    # Store the reweighter in a pickle dump
    d0_mc_corrections.weighter_dir().mkdir(exist_ok=True)

    dump_path = (
        d0_mc_corrections.pgun_path(year, "cf", magnetisation)
        if mc_type == "pgun"
        else d0_mc_corrections.mc_path(year, "cf", magnetisation)
    )
    with open(str(dump_path), "wb") as weighter_f:
        pickle.dump(weighter, weighter_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )
    parser.add_argument(
        "mc_type", type=str, help="full MC or pgun MC", choices={"pgun", "MC"}
    )

    main(**vars(parser.parse_args()))
