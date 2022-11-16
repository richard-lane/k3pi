"""
Read the (WS) real data dataframe,
apply BDT cut to it,
apply efficiency correction,
do the mass fit,
plot it

"""
import os
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars
from libFit import fit, pdfs
from lib_efficiency import efficiency_model
from lib_efficiency.efficiency_util import k_3pi
from lib_efficiency.metrics import _counts
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_time_fit.definitions import TIME_BINS


def _time_bin_indices(times: np.ndarray) -> np.ndarray:
    """Time bin indices"""
    bins = (MIN_TIME, *TIME_BINS[2:])
    return np.digitize(times, bins)


def _bdt_cut(dataframe: pd.DataFrame, year: str, magnetisation: str) -> pd.DataFrame:
    """
    Perform the BDT cut on a dataframe and return a slice
    that passed the cut

    """
    # Always use the DCS BDT to do cuts
    clf = get_clf(year, "dcs", magnetisation)
    training_labels = list(training_vars.training_var_names())
    threshhold = 0.185
    predicted_signal = clf.predict_proba(dataframe[training_labels])[:, 1] > threshhold

    print(f"{np.sum(predicted_signal)} of {len(predicted_signal)} predicted signal")
    return dataframe[predicted_signal]


def _efficiency_wts(
    dataframe: pd.DataFrame, year: str, sign: str, magnetisation: str
) -> np.ndarray:
    """
    Weights to account for efficiency effects

    uses reweighter trained on both

    """
    # Find weights using the API
    return efficiency_model.weights(
        *k_3pi(dataframe),
        dataframe["time"],
        "both",
        year,
        sign,
        magnetisation,
        fit=False,
        cut=True,
        verbose=True,
    )


def _plot_masses(
    axis: plt.Axes,
    dataframe: pd.DataFrame,
    bins: np.ndarray,
    /,
    weights: np.ndarray = None,
    label: str = None,
) -> None:
    """
    Plot histogram of Delta M

    """
    delta_m = dataframe["D* mass"] - dataframe["D0 mass"]

    axis.hist(delta_m, bins=bins, histtype="step", weights=weights, label=label)


def _plot_hists(
    before_df: pd.DataFrame,
    after_bdt_cut: pd.DataFrame,
    efficiency_wts: np.ndarray,
    bins: np.ndarray,
    path: str,
):
    """
    Plot histograms of Delta M before/after BDT cut and efficiency
    correction

    """
    fig, axis = plt.subplots()
    _plot_masses(axis, before_df, bins, weights=None, label="raw data")
    _plot_masses(axis, after_bdt_cut, bins, weights=None, label="After BDT cut")
    _plot_masses(
        axis,
        after_bdt_cut,
        bins,
        weights=efficiency_wts,
        label="After Efficiency Correction",
    )
    axis.set_xlabel(r"$\Delta M$")
    axis.legend()
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def _plot_fit(
    dcs_count: np.ndarray,
    cf_count: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    path: str,
    dcs_err: np.ndarray,
    cf_err: np.ndarray,
) -> None:
    """
    Plot the fit, assuming constant width bins

    """
    cf_params = (params[0], *params[2:])
    dcs_params = tuple(params[1:])

    def fitted_pdf(pts: np.ndarray, fit_params: Tuple) -> np.ndarray:
        return pdfs.fractional_pdf(pts, *fit_params)

    def fitted_sig(pts: np.ndarray, fit_params) -> np.ndarray:
        return fit_params[0] * pdfs.normalised_signal(pts, *fit_params[1:-2])

    def fitted_bkg(pts: np.ndarray, fit_params) -> np.ndarray:
        return (1 - fit_params[0]) * pdfs.normalised_bkg(pts, *fit_params[-2:])

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(14, 7)
    )
    centres = (bins[1:] + bins[:-1]) / 2

    if cf_err is None:
        cf_err = np.sqrt(cf_count)
    if dcs_err is None:
        dcs_err = np.sqrt(dcs_count)

    # Assumes constant width bins
    dcs_scale_factor = np.sum(dcs_count) * (bins[1] - bins[0])
    cf_scale_factor = np.sum(cf_count) * (bins[1] - bins[0])

    cf_predicted = cf_scale_factor * fitted_pdf(centres, cf_params)
    axes["A"].errorbar(centres, cf_count, yerr=cf_err, fmt="k.")
    axes["A"].plot(centres, cf_predicted, "r-")

    dcs_predicted = dcs_scale_factor * fitted_pdf(centres, dcs_params)
    axes["B"].errorbar(centres, dcs_count, yerr=dcs_err, fmt="k.")
    axes["B"].plot(centres, dcs_predicted, "r-")

    axes["A"].plot(
        centres,
        cf_scale_factor * fitted_sig(centres, cf_params),
        color="orange",
        label="Signal",
    )
    axes["B"].plot(
        centres,
        dcs_scale_factor * fitted_sig(centres, dcs_params),
        color="orange",
        label="Signal",
    )

    axes["A"].plot(
        centres,
        cf_scale_factor * fitted_bkg(centres, cf_params),
        color="blue",
        label="Bkg",
    )
    axes["B"].plot(
        centres,
        dcs_scale_factor * fitted_bkg(centres, dcs_params),
        color="blue",
        label="Bkg",
    )
    axes["A"].legend()

    dcs_diffs = dcs_count - dcs_predicted
    cf_diffs = cf_count - cf_predicted

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["C"].errorbar(centres, dcs_diffs, yerr=dcs_err, fmt="k.")
    axes["C"].set_yticklabels([])

    axes["D"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].errorbar(centres, cf_diffs, yerr=cf_err, fmt="k.")
    axes["D"].set_yticklabels([])

    fig.savefig(path)
    plt.close(fig)


def _make_plots(
    dcs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    dcs_bdt_cut: pd.DataFrame,
    cf_bdt_cut: pd.DataFrame,
    prefix: str,
    dcs_weights: np.ndarray,
    cf_weights: np.ndarray,
    time_bin: int,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

    """
    # Plot histograms of the masses
    bins = np.linspace(*pdfs.domain(), 200)
    _plot_hists(dcs_df, dcs_bdt_cut, dcs_weights, bins, f"{prefix}dcs_mass_hists.png")
    _plot_hists(cf_df, cf_bdt_cut, cf_weights, bins, f"{prefix}cf_mass_hists.png")

    # Mass fit before anything
    dcs_delta_m = dcs_df["D* mass"] - dcs_df["D0 mass"]
    cf_delta_m = cf_df["D* mass"] - cf_df["D0 mass"]
    dcs_count, dcs_err = _counts(dcs_delta_m, np.ones_like(dcs_delta_m), bins)
    cf_count, cf_err = _counts(cf_delta_m, np.ones_like(cf_delta_m), bins)
    _plot_fit(
        dcs_count,
        cf_count,
        bins,
        fit.binned_simultaneous_fit(cf_count, dcs_count, bins, time_bin).values,
        f"{prefix}before_cut.png",
        None,
        None,
    )

    # Mass fit after BDT cut
    dcs_delta_m = dcs_bdt_cut["D* mass"] - dcs_bdt_cut["D0 mass"]
    cf_delta_m = cf_bdt_cut["D* mass"] - cf_bdt_cut["D0 mass"]
    dcs_count, dcs_err = _counts(dcs_delta_m, dcs_weights, bins)
    cf_count, cf_err = _counts(cf_delta_m, cf_weights, bins)
    _plot_fit(
        dcs_count,
        cf_count,
        bins,
        fit.binned_simultaneous_fit(cf_count, dcs_count, bins, time_bin).values,
        f"{prefix}after_cut.png",
        None,
        None,
    )

    # After efficiency correction
    _plot_fit(
        dcs_count,
        cf_count,
        bins,
        fit.binned_simultaneous_fit(
            cf_count, dcs_count, bins, time_bin, rs_errors=cf_err, ws_errors=dcs_err
        ).values,
        f"{prefix}after_correction.png",
        dcs_err=dcs_err,
        cf_err=cf_err,
    )


def main():
    """
    Plot a real data mass fit after all my manipulations

    """
    # Read data
    year, magnetisation = "2018", "magdown"
    dcs_df = pd.concat(get.data(year, "dcs", magnetisation))
    cf_df = pd.concat(get.data(year, "cf", magnetisation))

    dcs_df = dcs_df[: len(dcs_df) // 2]
    cf_df = cf_df[: len(cf_df) // 2]

    # Find time bin indices
    dcs_indices = _time_bin_indices(dcs_df["time"])
    cf_indices = _time_bin_indices(cf_df["time"])

    # Do BDT cut with the right threshhold
    print("Doing BDT cut")
    dcs_bdt_cut = _bdt_cut(dcs_df, year, magnetisation)
    cf_bdt_cut = _bdt_cut(cf_df, year, magnetisation)
    dcs_cut_indices = _time_bin_indices(dcs_bdt_cut["time"])
    cf_cut_indices = _time_bin_indices(cf_bdt_cut["time"])

    # Do efficiency correction
    dcs_weights = _efficiency_wts(dcs_bdt_cut, year, "dcs", magnetisation)
    cf_weights = _efficiency_wts(cf_bdt_cut, year, "cf", magnetisation)

    # Plot stuff
    fit_dir = "fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    # Don't care about the first and last bins
    # (under/over flow)
    for index in tqdm(np.unique(dcs_indices)[1:-1]):
        _make_plots(
            dcs_df[dcs_indices == index],
            cf_df[cf_indices == index],
            dcs_bdt_cut[dcs_cut_indices == index],
            cf_bdt_cut[cf_cut_indices == index],
            f"{fit_dir}bin{index}_",
            dcs_weights[dcs_cut_indices == index],
            cf_weights[cf_cut_indices == index],
            index,
        )


if __name__ == "__main__":
    main()
