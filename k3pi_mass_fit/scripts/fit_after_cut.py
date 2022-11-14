"""
Read the (WS) real data dataframe,
apply BDT cut to it,
apply efficiency correction,
do the mass fit,
plot it

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars
from libFit import fit, pdfs
from lib_efficiency import efficiency_model
from lib_efficiency.efficiency_util import k_3pi
from lib_efficiency.metrics import _counts


def _plot(dataframe: pd.DataFrame, path: str):
    """
    Plot histograms of every variable in the dataframe

    """
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    hist_kw = {"histtype": "step"}
    for axis, label in zip(axes.ravel(), dataframe):
        # Find histogram bins
        low, high = np.quantile(dataframe[label], [0.05, 0.95])
        bins = np.linspace(0.9 * low, 1.1 * high, 100)

        # Plot
        axis.hist(dataframe[label], label="data", bins=bins, **hist_kw)

        axis.set_title(label)
        axis.legend()

    # fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


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


def _plot_fit(
    dcs_deltam: np.ndarray,
    cf_deltam: np.ndarray,
    params: Tuple,
    path: str,
    dcs_wt: np.ndarray,
    cf_wt: np.ndarray,
) -> None:
    """
    Plot the fit

    """
    cf_params = (params[0], *params[2:])
    dcs_params = params[1:]

    def fitted_pdf(pts: np.ndarray, params: Tuple) -> np.ndarray:
        return pdfs.fractional_pdf(pts, *params)

    def fitted_sig(pts: np.ndarray, params) -> np.ndarray:
        return pdfs.normalised_signal(pts, *params[1:-2])

    def fitted_bkg(pts: np.ndarray, params) -> np.ndarray:
        return pdfs.normalised_bkg(pts, *params[-2:])

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(14, 7)
    )

    bins = np.linspace(*pdfs.domain(), 250)
    centres = (bins[1:] + bins[:-1]) / 2

    if cf_wt is None:
        cf_wt = np.ones_like(cf_deltam)
    if dcs_wt is None:
        dcs_wt = np.ones_like(dcs_deltam)
    cf_counts, cf_err = _counts(cf_deltam, cf_wt, bins)
    dcs_counts, dcs_err = _counts(dcs_deltam, dcs_wt, bins)

    num_dcs = np.sum(dcs_wt)
    num_cf = np.sum(cf_wt)
    dcs_scale_factor = num_dcs * (bins[1] - bins[0])
    cf_scale_factor = num_cf * (bins[1] - bins[0])

    cf_predicted = cf_scale_factor * fitted_pdf(centres, cf_params)
    axes["A"].errorbar(centres, cf_counts, yerr=cf_err, fmt="k.")
    axes["A"].plot(centres, cf_predicted, "r-")

    dcs_predicted = dcs_scale_factor * fitted_pdf(centres, dcs_params)
    axes["B"].errorbar(centres, dcs_counts, yerr=dcs_err, fmt="k.")
    axes["B"].plot(centres, dcs_predicted, "r-")

    axes["A"].plot(
        centres,
        cf_scale_factor * cf_params[0] * fitted_sig(centres, cf_params),
        color="orange",
        label="Signal",
    )
    axes["B"].plot(
        centres,
        dcs_scale_factor * dcs_params[0] * fitted_sig(centres, dcs_params),
        color="orange",
        label="Signal",
    )

    axes["A"].plot(
        centres,
        dcs_scale_factor * (1 - dcs_params[0]) * fitted_bkg(centres, dcs_params),
        color="blue",
        label="Bkg",
    )
    axes["B"].plot(
        centres,
        cf_scale_factor * (1 - cf_params[0]) * fitted_bkg(centres, cf_params),
        color="blue",
        label="Bkg",
    )
    axes["A"].legend()

    dcs_diffs = dcs_counts - dcs_predicted
    cf_diffs = cf_counts - cf_predicted

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["C"].errorbar(centres, dcs_diffs, yerr=dcs_err, fmt="k.")
    axes["C"].set_yticklabels([])

    axes["D"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].errorbar(centres, cf_diffs, yerr=cf_err, fmt="k.")
    axes["D"].set_yticklabels([])

    fig.savefig(path)


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

    # Do BDT cut with the right threshhold
    print("Doing BDT cut")
    dcs_bdt_cut = _bdt_cut(dcs_df, year, magnetisation)
    cf_bdt_cut = _bdt_cut(cf_df, year, magnetisation)

    # Plot training vars
    _plot(dcs_df, "dcs_data_vars_all.png")
    _plot(cf_df, "cf_data_vars_all.png")
    _plot(dcs_bdt_cut, "dcs_data_vars_cut.png")
    _plot(cf_bdt_cut, "cf_data_vars_cut.png")

    # Do efficiency correction
    dcs_weights = _efficiency_wts(dcs_bdt_cut, year, "dcs", magnetisation)
    cf_weights = _efficiency_wts(cf_bdt_cut, year, "cf", magnetisation)

    # Plot stuff
    bins = np.linspace(*pdfs.domain(), 500)
    _plot_hists(dcs_df, dcs_bdt_cut, dcs_weights, bins, "dcs_mass_hists.png")
    _plot_hists(cf_df, cf_bdt_cut, cf_weights, bins, "cf_mass_hists.png")

    # Should also probably plot the mass fit as well
    dcs_delta_m = dcs_df["D* mass"] - dcs_df["D0 mass"]
    cf_delta_m = cf_df["D* mass"] - cf_df["D0 mass"]

    _plot_fit(
        dcs_delta_m,
        cf_delta_m,
        fit.binned_simultaneous_fit(cf_delta_m, dcs_delta_m, bins, 5).values,
        "before_cut.png",
        None,
        None,
    )

    dcs_delta_m = dcs_bdt_cut["D* mass"] - dcs_bdt_cut["D0 mass"]
    cf_delta_m = cf_bdt_cut["D* mass"] - cf_bdt_cut["D0 mass"]
    _plot_fit(
        dcs_delta_m,
        cf_delta_m,
        fit.binned_simultaneous_fit(cf_delta_m, dcs_delta_m, bins, 5).values,
        "after_cut.png",
        None,
        None,
    )

    # After efficiency correction
    _plot_fit(
        dcs_delta_m,
        cf_delta_m,
        fit.binned_simultaneous_fit(
            cf_delta_m, dcs_delta_m, bins, 5, rs_weights=None, ws_weights=dcs_weights
        ).values,
        "after_correction.png",
        dcs_wt=dcs_weights,
        cf_wt=cf_weights,
    )


if __name__ == "__main__":
    main()
