"""
Plot the cut efficiency for the testing sample

Ideally it will be constant in decay time

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def _plot_ratio(
    ax: plt.Axes,
    numerator: np.ndarray,
    denominator: np.ndarray,
    bins: np.ndarray,
    label: str,
    **plot_kw,
):
    """
    Plot binned ratio of two arrays on an axis

    :param ax: axes to plot on
    :param numerator: the variable we want to plot the efficiency in.
                      Efficiency is numerator/denominator counts in each bin
    :param numerator: the variable we want to plot the efficiency in
    :param bins: bins for variable x
    :param label: axis label for the efficiency

    """
    numerator_indices = np.digitize(numerator, bins)
    denominator_indices = np.digitize(denominator, bins)

    ratio, error = [], []
    n_bins = len(bins) - 1
    for i in range(1, n_bins + 1):
        n_num = np.sum(numerator_indices == i)
        n_denom = np.sum(denominator_indices == i)

        err_num = np.sqrt(n_num)
        err_denom = np.sqrt(n_denom)

        ratio.append(n_num / n_denom)
        error.append(
            ratio[-1] * np.sqrt((err_num / n_num) ** 2 + (err_denom / n_denom) ** 2)
        )

    ratio = np.array(ratio)
    error = np.array(error)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    ax.errorbar(
        centres,
        ratio,
        xerr=widths,
        yerr=error,
        fmt=".",
        label=label,
        **plot_kw,
    )


def _delta_m_eff(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """
    Cut efficiency in Delta M for bkg

    """
    delta_m = dataframe["D* mass"] - dataframe["D0 mass"]

    fig, ax = plt.subplot_mosaic("AAA\nAAA\nBBB")
    bins = np.linspace(152, 157, 15)
    _plot_ratio(
        ax["A"], delta_m[predictions == 1], delta_m, bins, "bkg", color="#ff7f0e"
    )

    ax["A"].legend()
    ax["A"].set_ylabel("efficiency")

    bins = np.linspace(bins[0], bins[-1], 100)
    ax["B"].hist(delta_m, bins, histtype="step", color="#ff7f0e")
    ax["B"].set_xlabel(r"$\Delta M$ / MeV")

    fig.suptitle("Cut efficiency")

    fig.savefig("delta_m_efficiency.png")
    plt.show()


def _dst_eff(
    sig_m: np.ndarray,
    bkg_m: np.ndarray,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
) -> None:
    """
    Cut efficiency with D* mass

    """
    fig, ax = plt.subplot_mosaic("AAA\nAAA\nBBB", sharex=True)
    bins = np.linspace(1930, 2100, 15)
    _plot_ratio(ax["A"], sig_m[sig_predictions == 1], sig_m, bins, "sig")
    _plot_ratio(ax["A"], bkg_m[bkg_predictions == 1], bkg_m, bins, "bkg")

    ax["A"].legend()
    ax["A"].set_ylabel("efficiency")

    ax["B"].set_xlabel(r"$M(D^*)$ / MeV")

    bins = np.linspace(bins[0], bins[-1], 100)
    ax["B"].hist(sig_m, bins=bins, histtype="step")
    ax["B"].hist(bkg_m, bins=bins, histtype="step")

    fig.suptitle("Cut efficiency")

    fig.savefig("dst_m_efficiency.png")
    plt.show()


def _d_eff(
    sig_m: np.ndarray,
    bkg_m: np.ndarray,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
) -> None:
    """
    Time cut efficiency

    """
    fig, ax = plt.subplot_mosaic("AAA\nAAA\nBBB", sharex=True)
    bins = np.linspace(1800, 1940, 15)
    _plot_ratio(ax["A"], sig_m[sig_predictions == 1], sig_m, bins, "sig")
    _plot_ratio(ax["A"], bkg_m[bkg_predictions == 1], bkg_m, bins, "bkg")

    ax["A"].legend()
    ax["A"].set_ylabel("efficiency")

    ax["B"].set_xlabel(r"$D^0M$ / MeV")

    bins = np.linspace(bins[0], bins[-1], 100)
    ax["B"].hist(sig_m, bins=bins, histtype="step")
    ax["B"].hist(bkg_m, bins=bins, histtype="step")

    fig.suptitle("Cut efficiency")

    fig.savefig("d0_m_efficiency.png")
    plt.show()


def _time_eff(
    sig_t: np.ndarray,
    bkg_t: np.ndarray,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
) -> None:
    """
    Time cut efficiency

    """
    fig, ax = plt.subplot_mosaic("AAA\nAAA\nBBB", sharex=True)
    bins = np.linspace(0, 8, 15)
    _plot_ratio(ax["A"], sig_t[sig_predictions == 1], sig_t, bins, "sig")
    _plot_ratio(ax["A"], bkg_t[bkg_predictions == 1], bkg_t, bins, "bkg")

    ax["A"].legend()

    fig.suptitle("Cut efficiency")

    bins = np.linspace(bins[0], bins[-1], 100)
    ax["B"].hist(sig_t, bins=bins, histtype="step")
    ax["B"].hist(sig_t, bins=bins, histtype="step")

    ax["B"].set_xlabel("t / ps")
    ax["A"].set_ylabel("efficiency")

    fig.savefig("time_efficiency.png")
    plt.show()


def main():
    """ Plot various efficiencies """
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = get_clf(year, sign, magnetisation)
    training_labels = list(training_vars.training_var_names())
    sig_predictions = clf.predict(sig_df[training_labels])
    bkg_predictions = clf.predict(bkg_df[training_labels])

    _time_eff(
        sig_df["time"],
        bkg_df["time"],
        sig_predictions,
        bkg_predictions,
    )

    _d_eff(
        sig_df["D0 mass"],
        bkg_df["D0 mass"],
        sig_predictions,
        bkg_predictions,
    )

    _dst_eff(
        sig_df["D* mass"],
        bkg_df["D* mass"],
        sig_predictions,
        bkg_predictions,
    )

    _delta_m_eff(
        bkg_df,
        bkg_predictions,
    )


if __name__ == "__main__":
    main()
