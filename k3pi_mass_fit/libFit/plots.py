import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from util import train_features


def plot_all(df: pd.DataFrame, predictions: np.ndarray, prefix: str) -> None:
    """
    Plot all columns in a dataframe

    :param df: dataframe
    :param predictions: whether events are predicted to be signal (1) or background (0)
    :param prefix: string prefix to prepend before the plot titles

    """
    signal = df["label_sig_bkg"] == 1
    bkg = df["label_sig_bkg"] == 0

    wt = df["nsig_sw"]

    # Masks for signal + background after performing cuts
    signal_after = (predictions == 1) & signal
    bkg_after = (predictions == 1) & bkg

    for col in df:
        fig, ax = plt.subplots(1, 2, sharey=True)
        x = df[col]
        # Plot true signal and background
        quantiles = np.quantile(x, [0.05, 0.95])
        bins = np.linspace(0.8 * quantiles[0], 1.2 * quantiles[1], 100)

        ax[0].hist(
            x[signal], bins=bins, histtype="step", label="signal", weights=wt[signal]
        )
        ax[0].hist(
            x[bkg], bins=bins, histtype="step", label="background", weights=wt[bkg]
        )
        ax[0].set_title("All")

        # Plot predicted signal and background
        ax[1].hist(
            x[signal_after],
            bins=bins,
            histtype="step",
            label="signal",
            weights=wt[signal_after],
        )
        ax[1].hist(
            x[bkg_after],
            bins=bins,
            histtype="step",
            label="background",
            weights=wt[bkg_after],
        )
        ax[1].set_title("Background Removed")

        for a in ax:
            a.legend()

        title = col if col not in train_features() else f"{col}*"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f"{prefix}{col}.png")
        plt.close(fig)


def roc(ax, proba, true, label):
    """
    ROC curve

    User should call ax.legend() later to draw legend

    Usage:
        roc(
            ax, clf.predict_proba(X)[:, -1], trainY, "my roc"
        )

    """
    fpr, tpr, _ = roc_curve(true, proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.4f}")


def plot_ratio(
    ax: plt.Axes,
    numerator: np.ndarray,
    denominator: np.ndarray,
    bins: np.ndarray,
    label: str,
    numerator_weights: np.ndarray = None,
    denominator_weights: np.ndarray = None,
) -> np.float64:
    """
    Plot binned ratio of two arrays on an axis

    Scales ratio to have a mean of 1

    Returns chi2 wrt ones

    :param ax: axes to plot on
    :param numerator: the variable we want to plot the efficiency in. Efficiency is numerator/denominator counts in each bin
    :param numerator: the variable we want to plot the efficiency in
    :param bins: bins for variable x
    :param label: axis label for the efficiency
    :param numerator_weights: weights to apply to numerator
    :param denominerator_weights: weights to apply to denominator

    :returns: chi2 comparing the ratio to an array of ones

    """
    if numerator_weights is None:
        numerator_weights = np.ones_like(numerator)
    if denominator_weights is None:
        denominator_weights = np.ones_like(denominator)

    numerator_indices = np.digitize(numerator, bins)
    denominator_indices = np.digitize(denominator, bins)

    ratio, error = [], []

    n_bins = len(bins) - 1
    for i in range(1, n_bins + 1):
        this_bin_num_wt = numerator_weights[numerator_indices == i]
        this_bin_denom_wt = denominator_weights[denominator_indices == i]

        n_num = np.sum(this_bin_num_wt)
        n_denom = np.sum(this_bin_denom_wt)

        err_num = np.sqrt(np.sum(this_bin_num_wt**2))
        err_denom = np.sqrt(np.sum(this_bin_denom_wt**2))

        ratio.append(n_num / n_denom)
        error.append(
            ratio[-1] * np.sqrt((err_num / n_num) ** 2 + (err_denom / n_denom) ** 2)
        )

    ratio = np.array(ratio)
    error = np.array(error)

    error /= np.mean(ratio)
    ratio /= np.mean(ratio)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    ax.errorbar(
        centres,
        ratio,
        xerr=widths,
        yerr=error,
        fmt=".",
        label=label,
    )

    return np.sum((ratio - np.ones_like(ratio)) ** 2 / (error**2))
