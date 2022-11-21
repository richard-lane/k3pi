"""
Read the (WS) real data dataframe,
apply BDT cut to it,
do both individual and simultaneous RS/WS mass fits,
plot them

"""
import os
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_data import get
from libFit import fit, pdfs, util
from lib_cuts.get import cut_dfs, classifier as get_clf


def fitted_pdf(pts: np.ndarray, fit_params: Tuple) -> np.ndarray:
    """Fitted sig + bkg pdf"""
    return pdfs.fractional_pdf(pts, *fit_params)


def fitted_sig(pts: np.ndarray, fit_params: Tuple) -> np.ndarray:
    """Fitted sig pdf"""
    return fit_params[0] * pdfs.normalised_signal(pts, *fit_params[1:-2])


def fitted_bkg(pts: np.ndarray, fit_params: Tuple) -> np.ndarray:
    """Fitted bkg pdf"""
    return (1 - fit_params[0]) * pdfs.normalised_bkg(pts, *fit_params[-2:])


def _plot(axis: plt.Axes, bins: np.ndarray, count: np.ndarray, params: Tuple) -> None:
    """Plot points and fit on an axis"""
    centres = (bins[1:] + bins[:-1]) / 2

    # Assume equal width bins
    scale_factor = np.sum(count) * (bins[1] - bins[0])

    axis.errorbar(centres, count, yerr=np.sqrt(count), fmt="k.")

    axis.plot(
        centres,
        scale_factor * fitted_sig(centres, params),
        color="orange",
        label="Signal",
    )
    axis.plot(
        centres,
        scale_factor * fitted_bkg(centres, params),
        color="blue",
        label="Background",
    )
    axis.plot(
        centres,
        scale_factor * fitted_pdf(centres, params),
        color="red",
        label="Fit",
    )


def _plot_diff(
    axis: plt.Axes, bins: np.ndarray, count: np.ndarray, params: Tuple
) -> None:
    """Plot diff between counts and fit on an axis"""
    centres = (bins[1:] + bins[:-1]) / 2

    # Assume equal width bins
    scale_factor = np.sum(count) * (bins[1] - bins[0])
    predicted = scale_factor * fitted_pdf(centres, params)

    diffs = count - predicted

    axis.plot(pdfs.domain(), [1, 1], "r-")
    axis.errorbar(centres, diffs, yerr=np.sqrt(count), fmt="k.")
    axis.set_yticklabels([])


def _simultaneous_fit(
    cf_count: np.ndarray, dcs_count: np.ndarray, bins: np.ndarray, fit_dir: str
) -> None:
    """
    Plot simultaneous fit

    """
    # Central time bin
    time_bin = 5

    # Do the fit
    params = fit.binned_simultaneous_fit(cf_count, dcs_count, bins, time_bin).values
    cf_params = (params[0], *params[2:])
    dcs_params = tuple(params[1:])

    # Plot histograms and fit
    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(14, 7)
    )
    _plot(axes["A"], bins, cf_count, cf_params)
    _plot(axes["B"], bins, dcs_count, dcs_params)

    # Plot pulls
    _plot_diff(axes["C"], bins, cf_count, cf_params)
    _plot_diff(axes["D"], bins, dcs_count, dcs_params)

    # Set the title to the yields
    # This is a little wasteful as it does the fit again, but maybe it's a good test
    # that the yields function is consistent or something
    (rs_yield, ws_yield), (rs_err, ws_err) = fit.yields(
        cf_count, dcs_count, bins, time_bin
    )
    axes["A"].set_title(rf"Yield: {rs_yield:,.2f}$\pm${rs_err:,.2f}")
    axes["B"].set_title(rf"Yield: {ws_yield:,.2f}$\pm${ws_err:,.2f}")

    path = f"{fit_dir}all_simultaneous.png"
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)


def _fit(count: np.ndarray, bins: np.ndarray, sign: str, path: str) -> None:
    """
    Plot individual fit

    """
    # Central time bin
    time_bin = 5

    # Do the fit
    sig_frac_guess = 0.95 if sign == "RS" else 0.05

    fitter = fit.binned_fit(count, bins, sign, time_bin, sig_frac_guess)
    params = fitter.values
    errors = fitter.errors

    # Plot histograms and fit
    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(14, 7))
    _plot(axes["A"], bins, count, params)

    # Plot pulls
    _plot_diff(axes["B"], bins, count, params)

    # Set the title to the yields
    axes["A"].set_title(
        rf"Yield: {params[0]* np.sum(count):,.2f}$\pm${errors[0] * np.sum(count):,.2f}"
    )

    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)


def _make_plots(
    dcs_count: pd.DataFrame,
    cf_count: pd.DataFrame,
    bins: np.ndarray,
    fit_dir: str,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

    """
    _simultaneous_fit(cf_count, dcs_count, bins, fit_dir)
    _fit(cf_count, bins, "RS", f"{fit_dir}cf.png")
    _fit(dcs_count, bins, "WS", f"{fit_dir}dcs.png")


def _counts(
    year: str, magnetisation: str, bins: np.ndarray, *, bdt_cut: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DCS and CF counts

    """
    dcs_gen = get.data(year, "dcs", magnetisation)
    cf_gen = get.data(year, "cf", magnetisation)

    if bdt_cut:
        clf = get_clf(year, "dcs", magnetisation)
        dcs_gen = cut_dfs(dcs_gen, clf)
        cf_gen = cut_dfs(cf_gen, clf)

    dcs_count, _ = util.delta_m_counts(dcs_gen, bins)
    cf_count, _ = util.delta_m_counts(cf_gen, bins)

    return dcs_count, cf_count


def main():
    """
    Plot a real data mass fit after all my manipulations

    """
    bins = np.linspace(*pdfs.domain(), 200)

    # Get delta M values from generator of dataframes
    year, magnetisation = "2018", "magdown"

    # Get delta M values from generator of dataframes after BDT cut
    # Do BDT cut with the right threshhold
    # dcs_bdt_cut = _bdt_cut(dcs_df, year, magnetisation)
    # cf_bdt_cut = _bdt_cut(cf_df, year, magnetisation)
    # dcs_cut_indices = _time_bin_indices(dcs_bdt_cut["time"])
    # cf_cut_indices = _time_bin_indices(cf_bdt_cut["time"])

    # Plot stuff
    fit_dir = "raw_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    _make_plots(
        *_counts(year, magnetisation, bins, bdt_cut=False),
        bins,
        f"{fit_dir}",
    )

    fit_dir = "bdt_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    _make_plots(
        *_counts(year, magnetisation, bins, bdt_cut=True),
        bins,
        f"{fit_dir}",
    )


if __name__ == "__main__":
    main()
