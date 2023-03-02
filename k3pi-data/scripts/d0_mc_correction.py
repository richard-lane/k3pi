"""
Make histograms of D0 eta and momentum;
show them and save to a dump somewhere

"""
import sys
import pathlib
import argparse
from typing import Tuple, Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from lib_data import get, d0_mc_corrections
from lib_cuts import get as cut_get
from libFit import fit, pdfs, definitions, plotting, sweighting, util as mass_util


def _massfit(
    dataframes: Iterable[pd.DataFrame],
    time_range: Tuple[float, float],
) -> Tuple[Tuple, np.ndarray, np.ndarray]:
    """
    Perform a massfit to CF dataframes without BDT cut

    Returns params, bins, fit range, counts

    """
    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()

    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], 144.5, 146.5, high), (n_underflow, 100, 200, 100)
    )

    counts, _ = mass_util.delta_m_counts(
        dataframes,
        mass_bins,
    )

    total = np.sum(counts)
    initial_guess = (
        0.9 * total,
        0.1 * total,
        *mass_util.signal_param_guess(),
        *mass_util.sqrt_bkg_param_guess("cf"),
    )

    fitter = fit.binned_fit(
        counts[n_underflow:],
        mass_bins[n_underflow:],
        initial_guess,
        fit_range,
        errors=None,  # Unweighted; assume Poisson errors
    )
    assert fitter.valid

    return fitter.values, mass_bins, fit_range, counts


def _sweights(
    year: str, magnetisation: str
) -> Tuple[Callable, plt.Figure, Iterable[plt.Axes]]:
    """
    Find a unary function that projects signal out, given
    delta M

    Uses CF for the sWeighting

    Also plots information relating to the mass fit and returns the figure and axes

    """
    # Keep all the times for now
    time_range = (-np.inf, np.inf)

    # Don't want the BDT cut here,
    # since the corrections are used as input for the bdt cut
    cf_dataframes = cut_get.time_cut_dfs(
        year, magnetisation, "cf", bdt_cut=False, time_range=time_range
    )

    # Do the mass fit, get parameters and mass fit counts
    fit_values, mass_bins, fit_range, counts = _massfit(cf_dataframes, time_range)

    # Plot the fit on Axes
    fig, axes = plt.subplot_mosaic("AAACC\nAAACC\nAAADD\nBBBDD", figsize=(12, 12))
    plotting.mass_fit(
        (axes["A"], axes["B"]),
        counts,
        np.sqrt(counts),
        mass_bins,
        fit_range,
        fit_values,
    )

    # Find the sWeighting function
    sweight_fcn = sweighting.signal_weight_fcn(fit_values, fit_range)

    # Return the function, figure and axes
    return sweight_fcn, fig, axes


def main(*, year: str, magnetisation: str):
    """
    Get particle gun, MC and data dataframes
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    # Find sWeights
    sweight_fcn, fig, axes = _sweights(year, magnetisation)

    # Plot mass fit from sWeighting, the sWeighted RS distributions

    # pgun_pts = d0_mc_corrections.d0_points(get.particle_gun(sign))
    # mc_pts = d0_mc_corrections.d0_points(get.mc(year, sign, magnetisation))
    # data_pts = d0_mc_corrections.d0_points(get.data(year, sign, magnetisation))

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

    main(**vars(parser.parse_args()))
