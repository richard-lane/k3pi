"""
Read the (WS) real data dataframe,
apply BDT cut to it,
apply efficiency correction,
do the mass fit,
plot it

"""
import sys
import pathlib
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


def _bdt_cut(
    dataframe: pd.DataFrame, year: str, sign: str, magnetisation: str
) -> pd.DataFrame:
    """
    Perform the BDT cut on a dataframe and return a slice
    that passed the cut

    """
    clf = get_clf(year, sign, magnetisation)
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
    before_df: pd.DataFrame, after_bdt_cut: pd.DataFrame, efficiency_wts: np.ndarray
):
    """
    Plot histograms of Delta M before/after BDT cut and efficiency
    correction

    """
    fig, axis = plt.subplots()
    bins = np.linspace(139, 152, 100)
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

    fig.savefig("mass_hists.png")


def _plot_fit(delta_m: np.ndarray, params: tuple, path: str) -> None:
    """
    Plot the fit

    """
    print(params[0])

    def fitted_pdf(pts: np.ndarray) -> np.ndarray:
        return pdfs.fractional_pdf(pts, *params)

    def fitted_sig(pts: np.ndarray) -> np.ndarray:
        return pdfs.normalised_signal(pts, *params[1:-2])

    def fitted_bkg(pts: np.ndarray) -> np.ndarray:
        return pdfs.normalised_bkg(pts, *params[-2:])

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(10, 10))

    bins = np.linspace(*pdfs.domain(), 250)
    centres = (bins[1:] + bins[:-1]) / 2

    counts, _ = np.histogram(delta_m, bins)
    err = np.sqrt(counts)
    num_evts = len(delta_m)
    scale_factor = num_evts * (bins[1] - bins[0])

    axes["A"].errorbar(centres, counts, yerr=err, fmt="k.")
    predicted_counts = scale_factor * fitted_pdf(centres)
    axes["A"].plot(centres, predicted_counts, "r-")

    axes["A"].plot(
        centres,
        scale_factor * params[0] * fitted_sig(centres),
        color="orange",
        label="Signal",
    )
    axes["A"].plot(
        centres,
        scale_factor * (1 - params[0]) * fitted_bkg(centres),
        color="blue",
        label="Bkg",
    )
    axes["A"].legend()

    diffs = counts - predicted_counts
    axes["B"].plot(pdfs.domain(), [1, 1], "r-")
    axes["B"].errorbar(centres, diffs, yerr=err, fmt="k.")
    axes["B"].set_yticklabels([])

    fig.savefig(path)


def main():
    """
    Plot a real data mass fit after all my manipulations

    """
    # Read data
    year, sign, magnetisation = "2018", "dcs", "magdown"
    dataframe = pd.concat(get.data(year, sign, magnetisation))

    # Do BDT cut with the right threshhold
    bdt_cut_df = _bdt_cut(dataframe, year, sign, magnetisation)

    # Plot training vars
    _plot(dataframe, "data_vars_all.png")
    _plot(bdt_cut_df, "data_vars_cut.png")

    # Do efficiency correction
    weights = _efficiency_wts(bdt_cut_df, year, sign, magnetisation)

    # Plot stuff
    _plot_hists(dataframe, bdt_cut_df, weights)

    # Do a mass fit to the thing we just removed background from with the BDT

    # Should also probably plot the mass fit as well
    delta_m = dataframe["D* mass"] - dataframe["D0 mass"]
    _plot_fit(delta_m, fit.fit(delta_m, "WS", 5, 0.2).values, "before_cut.png")

    delta_m = bdt_cut_df["D* mass"] - bdt_cut_df["D0 mass"]
    _plot_fit(delta_m, fit.fit(delta_m, "WS", 5, 0.2).values, "after_cut.png")


if __name__ == "__main__":
    main()
