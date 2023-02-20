"""
Plot mass fits

"""
import sys
import pathlib
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
from lib_data import stats
from . import pdfs, util, bkg


def mass_fit(
    axes: Tuple[plt.Axes, plt.Axes],
    counts: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    fit_range: Tuple[float, float],
    fit_params: Tuple,
) -> None:
    """
    Plot the mass fit and pulls on an axis

    :param axes: tuple of (histogram axis, pull axis)
    :param counts: counts in each bin
    :param errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_range: domain used in the fit.
    :param fit_params: fit parameters; (sig frac, bkg frac, other params)

    """
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Which points are in the fit region
    # Assumes the fit range covers all the points but the lowest
    # ones
    fit = centres > fit_range[0]

    # Plot histogram
    err_kw = {"fmt": "k.", "elinewidth": 0.5, "markersize": 1.0}
    axes[0].errorbar(
        centres,
        counts / bin_widths,
        xerr=bin_widths / 2,  # Error bar is half width
        yerr=errs / bin_widths,
        **err_kw,
    )

    predicted = stats.areas(bins, pdfs.model(bins, *fit_params, fit_range)) / bin_widths

    predicted_signal = (
        fit_params[0]
        * stats.areas(bins, pdfs.normalised_signal(bins, *fit_params[2:-2], fit_range))
        / bin_widths
    )
    predicted_bkg = (
        fit_params[1]
        * stats.areas(bins, pdfs.normalised_bkg(bins, *fit_params[-2:], fit_range))
        / bin_widths
    )

    # Plot in the fit region
    axes[0].plot(centres[fit], predicted[fit], color="b")
    axes[0].plot(centres[fit], predicted_signal[fit], label="signal", color="orange")
    axes[0].plot(centres[fit], predicted_bkg[fit], label="bkg", color="green")

    # Plot the whole region
    axes[0].plot(centres, predicted, color="b", linestyle="--")
    axes[0].plot(
        centres,
        predicted_signal,
        color="orange",
        linestyle="--",
    )
    axes[0].plot(
        centres,
        predicted_bkg,
        color="green",
        linestyle="--",
    )

    axes[0].legend()

    # Plot pull
    diff = ((counts / bin_widths) - predicted) / (errs / bin_widths)
    axes[1].axhline(0, color="k")
    for pos in (-1, 1):
        axes[1].axhline(pos, color="r", alpha=0.5)

    # Only plot pull for the fit values
    axes[1].errorbar(
        centres[fit],
        diff[fit],
        yerr=1.0,
        **err_kw,
    )
    axes[0].set_ylabel("N/MeV")
    axes[1].set_ylabel(r"$\sigma$")
    axes[1].set_xlabel(r"$\Delta M$ /MeV")


def simul_fits(
    rs_counts: np.ndarray,
    rs_errs: np.ndarray,
    ws_counts: np.ndarray,
    ws_errs: np.ndarray,
    bins: np.ndarray,
    fit_range: Tuple[float, float],
    fit_params: Tuple,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Plot a simultaneous RS and WS fit on four Axes - two histograms
    and two pulls

    :param rs_counts: counts in each bin
    :param rs_errs: errors on the counts in each bin
    :param ws_counts: counts in each bin
    :param ws_errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_params: fit parameters as returned by the simultaneous fitter

    :returns: the figure
    :returns: a dict of A/B/C/D and the plot axes

    """
    rs_params, ws_params = util.rs_ws_params(fit_params)

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(12, 8)
    )
    mass_fit((axes["A"], axes["C"]), rs_counts, rs_errs, bins, fit_range, rs_params)
    mass_fit((axes["B"], axes["D"]), ws_counts, ws_errs, bins, fit_range, ws_params)

    axes["A"].legend()

    fig.tight_layout()

    return fig, axes


def alt_bkg_fit(
    axes: Tuple[plt.Axes, plt.Axes],
    counts: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    fit_params: Tuple,
    *,
    sign: str,
    bdt_cut: bool,
    efficiency: bool,
) -> None:
    """
    Plot the mass fit and pulls on an axis, using the alt bkg model

    Assumes the bins have equal widths

    :param axes: tuple of (histogram axis, pull axis)
    :param counts: counts in each bin
    :param errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_params: fit parameters; (sig frac, bkg frac, other params)
    :param: sign "cf" or "dcs", for finding the right estimated bkg pickle
    :param: bdt_cut for finding the right estimated bkg pickle
    :param: efficiency for finding the right estimated bkg pickle

    """
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Plot histogram
    err_kw = {"fmt": "k.", "elinewidth": 0.5, "markersize": 1.0}
    axes[0].errorbar(
        centres,
        counts,
        yerr=errs,
        **err_kw,
    )

    bkg_pdf = bkg.pdf(100, sign=sign, bdt_cut=bdt_cut, efficiency=efficiency)
    params = (*fit_params[:8], bkg_pdf, *fit_params[8:])
    predicted = bin_widths * pdfs.model_alt_bkg(centres, *params)

    predicted_signal = fit_params[0] * stats.areas(
        bins, pdfs.normalised_signal(bins, *fit_params[2:-3])
    )
    predicted_bkg = (
        fit_params[1]
        * bin_widths
        * pdfs.estimated_bkg(centres, bkg_pdf, *fit_params[-3:])
    )

    axes[0].plot(centres, predicted)
    axes[0].plot(
        centres,
        predicted_signal,
        label="signal",
    )
    axes[0].plot(
        centres,
        predicted_bkg,
        label="bkg",
    )

    # Plot pull
    diff = counts - predicted
    axes[1].plot(pdfs.domain(), [1, 1], "r-")
    axes[1].errorbar(
        centres,
        diff,
        yerr=errs,
        **err_kw,
    )


def alt_bkg_simul(
    rs_counts: np.ndarray,
    rs_errs: np.ndarray,
    ws_counts: np.ndarray,
    ws_errs: np.ndarray,
    bins: np.ndarray,
    fit_params: Tuple,
    *,
    bdt_cut: bool,
    efficiency: bool,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Plot a simultaneous RS and WS fit on four Axes - two histograms
    and two pulls

    :param rs_counts: counts in each bin
    :param rs_errs: errors on the counts in each bin
    :param ws_counts: counts in each bin
    :param ws_errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_params: fit parameters as returned by the simultaneous fitter

    :returns: the figure
    :returns: a dict of A/B/C/D and the plot axes

    """
    rs_params, ws_params = util.alt_rs_ws_params(fit_params)

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(12, 8)
    )

    alt_bkg_fit(
        (axes["A"], axes["C"]),
        rs_counts,
        rs_errs,
        bins,
        rs_params,
        sign="cf",
        bdt_cut=bdt_cut,
        efficiency=efficiency,
    )
    alt_bkg_fit(
        (axes["B"], axes["D"]),
        ws_counts,
        ws_errs,
        bins,
        ws_params,
        sign="dcs",
        bdt_cut=bdt_cut,
        efficiency=efficiency,
    )

    axes["A"].legend()

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].plot(pdfs.domain(), [1, 1], "r-")

    fig.tight_layout()

    return fig, axes
