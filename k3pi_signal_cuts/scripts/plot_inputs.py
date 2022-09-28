"""
Plot some relevant variables used for selecting the data/training the classifier

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import get, training_vars
from lib_cuts import util


def _plot(
    axis: plt.Axes, signal: np.ndarray, bkg: np.ndarray, sig_wt: np.ndarray
) -> None:
    """
    Plot signal + bkg on an axis

    """
    hist_kw = {"histtype": "step"}
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100

    sig_quantile = np.quantile(signal, quantiles)
    bkg_quantile = np.quantile(bkg, quantiles)

    bins = np.linspace(
        min(bkg_quantile[0], sig_quantile[0]),
        max(bkg_quantile[1], sig_quantile[1]),
        n_bins,
    )

    axis.hist(
        signal,
        bins=bins,
        **hist_kw,
        label="sig",
        color="b",
        weights=sig_wt * np.ones_like(signal)
    )
    axis.hist(bkg, bins=bins, **hist_kw, label="bkg", color="r")

    axis.legend()


def main():
    """
    Plot the training vars and some masses

    """
    year, magnetisation = "2018", "magdown"

    # Read the right data
    mc_df = get.mc(year, "dcs", magnetisation)
    uppermass_df = pd.concat(get.uppermass(year, "dcs", magnetisation))

    # Scale
    # The first weight will be the weight we want to apply to all signal events
    sig_frac = 0.0852
    sig_wt = util.weights(
        np.concatenate((np.ones(len(mc_df)), np.zeros(len(uppermass_df)))), sig_frac
    )[0]

    # Plot
    fig, ax = plt.subplots(4, 4, figsize=(15, 9))

    columns = list(training_vars.training_var_names()) + ["D0 mass", "D* mass"]
    for col, axis in zip(columns, ax.ravel()):
        _plot(axis, mc_df[col], uppermass_df[col], sig_wt)

        title = col if col in training_vars.training_var_names() else col + "*"
        axis.set_title(title)

    # Let's also plot the mass difference
    _plot(
        ax.ravel()[-1],
        mc_df["D* mass"] - mc_df["D0 mass"],
        uppermass_df["D* mass"] - uppermass_df["D0 mass"],
        sig_wt,
    )
    ax.ravel()[-1].set_title(r"$\Delta$M*")

    fig.tight_layout()

    fig.savefig("clf_cut_inputs.png")

    plt.show()


if __name__ == "__main__":
    main()
