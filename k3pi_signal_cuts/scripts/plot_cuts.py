"""
Plot testing data variables before and after cuts

"""
import sys
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import util, definitions
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars, cuts, d0_mc_corrections
from libFit.util import delta_m


def _plot(
    axis: plt.Axes,
    signal: np.ndarray,
    bkg: np.ndarray,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
    mc_corr_wts,
) -> None:
    """
    Plot signal + bkg on an axis

    """
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100

    sig_quantile = np.quantile(signal, quantiles)
    bkg_quantile = np.quantile(bkg, quantiles)

    bins = np.linspace(
        min(bkg_quantile[0], sig_quantile[0]),
        max(bkg_quantile[1], sig_quantile[1]),
        n_bins,
    )

    # Plot before cuts
    hist_kw = {"bins": bins}
    axis.hist(
        signal, label="sig", histtype="step", color="b", **hist_kw, weights=mc_corr_wts
    )
    axis.hist(bkg, label="bkg", histtype="step", color="r", **hist_kw)

    # Plot after cuts
    axis.hist(
        signal[sig_predictions == 1],
        label="sig",
        alpha=0.5,
        color="b",
        **hist_kw,
        weights=mc_corr_wts[sig_predictions == 1],
    )
    axis.hist(bkg[bkg_predictions == 1], label="bkg", alpha=0.5, color="r", **hist_kw)

    # Plot after cuts without MC correction
    # hist_kw = {**hist_kw, "linestyle": ":", "color": "k", "histtype": "step"}
    # axis.hist(
    #     signal,
    #     **hist_kw,
    # )
    # axis.hist(
    #     signal[sig_predictions == 1],
    #     **hist_kw,
    # )


def main(*, year: str, sign: str, magnetisation: str):
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    mc_corr_wts = d0_mc_corrections.mc_wt_df(sig_df, year, magnetisation)

    # Predict which of these are signal and background using our classifier
    clf = get_clf(year, sign, magnetisation)

    training_labels = list(training_vars.training_var_names())

    # Lets also undersample so we get the same amount of signal/bkg that we expect to see
    # in the data
    gen = np.random.default_rng()
    bkg_df = util.discard(gen, bkg_df, definitions.EXPECTED_N_BKG_SIG_REGION)

    sig_mask = gen.random(len(sig_df)) < (
        definitions.EXPECTED_N_SIG_SIG_REGION / len(sig_df)
    )
    sig_df = sig_df[sig_mask]
    mc_corr_wts = mc_corr_wts[sig_mask]

    threshhold = definitions.THRESHOLD
    sig_predictions = clf.predict_proba(sig_df[training_labels])[:, 1] > threshhold
    bkg_predictions = clf.predict_proba(bkg_df[training_labels])[:, 1] > threshhold

    # Plot histograms of our variables before/after doing these cuts
    columns = list(training_vars.training_var_names()) + ["Dst_ReFit_D0_M", "D* mass"]
    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    for col, axis in zip(columns, ax.ravel()):
        _plot(
            axis,
            sig_df[col],
            bkg_df[col],
            sig_predictions,
            bkg_predictions,
            mc_corr_wts,
        )

        title = col if col in training_vars.training_var_names() else col + "*"
        axis.set_title(title)

    # Let's also plot the mass difference
    _plot(
        ax.ravel()[-1],
        delta_m(sig_df),
        delta_m(bkg_df),
        sig_predictions,
        bkg_predictions,
        mc_corr_wts,
    )
    ax.ravel()[-1].set_title(r"$\Delta$M*")

    ax[0, 0].legend()

    fig.tight_layout()

    path = f"cuts_{year}_{sign}_{magnetisation}.png"
    plt.savefig(path)
    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, ax), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histograms showing cuts for simulation data (testing sample)"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(**vars(parser.parse_args()))
