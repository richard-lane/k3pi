"""
Train a classifier to separate signal and background

You should have already downloaded the data files and created pickle dumps

"""
import os
import sys
import pickle
import pathlib
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report

from lib_cuts import definitions
from lib_cuts import util

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "k3pi_mass_fit"))

from lib_data import get, training_vars, d0_mc_corrections, cuts
from libFit.util import delta_m


def _train_test_dfs(year, sign, magnetisation):
    """
    Join signal + bkg dataframes but split by train/test

    Returns also weights for the MC correction
    """
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(
        list(cuts.ipchi2_cut_dfs(get.uppermass(year, sign, magnetisation)))
    )

    # Add delta M column cus its easier
    sig_df["delta M"] = delta_m(sig_df)
    bkg_df["delta M"] = delta_m(bkg_df)

    # Find MC correction wts for the sig df
    # Scale st their average is 1.0
    # TODO put this back
    # mc_corr_wts = d0_mc_corrections.mc_weights(year, sign, magnetisation)
    # mc_corr_wts /= np.mean(mc_corr_wts)
    mc_corr_wts = np.ones(len(sig_df))

    combined_df = pd.concat((sig_df, bkg_df))

    combined_wts = np.concatenate((mc_corr_wts, np.ones(len(bkg_df))))

    # 1 for signal, 0 for background
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    train_mask = combined_df["train"]
    return (
        combined_df[train_mask],
        combined_df[~train_mask],
        labels[train_mask],
        labels[~train_mask],
        combined_wts[train_mask],
        combined_wts[~train_mask],
    )


def _plot_masses(
    train_df: pd.DataFrame, train_label: np.ndarray, weights: np.ndarray, path: str
) -> None:
    """
    Saves to path

    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    bkg_df = train_df[train_label == 0]
    sig_df = train_df[train_label == 1]

    bkg_wt = weights[train_label == 0]
    sig_wt = weights[train_label == 1]

    ax[0].hist(
        sig_df["Dst_ReFit_D0_M"],
        bins=np.linspace(1775, 1950, 200),
        color="b",
        alpha=0.4,
        weights=sig_wt,
    )
    ax[0].hist(
        bkg_df["Dst_ReFit_D0_M"],
        bins=np.linspace(1775, 1950, 200),
        color="r",
        alpha=0.4,
        weights=bkg_wt,
    )

    ax[1].hist(
        sig_df["delta M"],
        bins=np.linspace(140, 160, 200),
        color="b",
        alpha=0.4,
        label="sig",
        weights=sig_wt,
    )
    ax[1].hist(
        bkg_df["delta M"],
        bins=np.linspace(140, 160, 200),
        color="r",
        alpha=0.4,
        label="bkg",
        weights=bkg_wt,
    )

    ax[1].legend()

    ax[0].set_title(r"$D^0$ Mass")
    ax[1].set_title(r"$\Delta M$")

    ax[0].set_xlabel("MeV")
    ax[1].set_xlabel("MeV")

    fig.tight_layout()

    fig.savefig(path)
    print(f"plotted {path}")


def _plot_train_vars(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    wts: np.ndarray,
    mc_corr_wts: np.ndarray,
) -> None:
    """
    Plot histograms of the training variables

    """
    sig_df, bkg_df = dataframe[labels == 1], dataframe[labels == 0]
    sig_wts, bkg_wts = wts[labels == 1], wts[labels == 0]
    sig_mc_wts, bkg_mc_wts = mc_corr_wts[labels == 1], mc_corr_wts[labels == 0]

    columns = list(training_vars.training_var_names()) + [
        "Dst_ReFit_D0_M",
        "D* mass",
        "delta M",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    bkg_kw = {"color": "r"}
    sig_kw = {"color": "b"}

    for col, axis in tqdm(zip(columns, axes.ravel()), total=len(axes.ravel())):
        sig = sig_df[col]
        bkg = bkg_df[col]

        # Find binning to use
        quantiles = [0.01, 0.99]
        sig_quantile = np.quantile(sig, quantiles)
        bkg_quantile = np.quantile(bkg, quantiles)
        bins = np.linspace(
            min(bkg_quantile[0], sig_quantile[0]),
            max(bkg_quantile[1], sig_quantile[1]),
            100,
        )

        # Plot both raw - shaded, translucent
        axis.hist(sig, bins=bins, label="Sig (raw)", **bkg_kw, alpha=0.5)
        axis.hist(bkg, bins=bins, label="Bkg (raw)", **sig_kw, alpha=0.5)

        # Plot both after stat reweighting - solid line
        axis.hist(
            sig,
            bins=bins,
            label="Sig (stat weighted)",
            color="k",
            histtype="step",
            weights=sig_wts,
        )
        axis.hist(
            bkg,
            bins=bins,
            label="Bkg (stat weighted)",
            color="k",
            histtype="step",
            weights=bkg_wts,
        )

        # Plot both after MC correction - dashed line
        axis.hist(
            sig,
            bins=bins,
            label="Sig (MC corrected)",
            **bkg_kw,
            histtype="step",
            weights=sig_wts * sig_mc_wts,
            linestyle=":",
        )
        axis.hist(
            bkg,
            bins=bins,
            label="Bkg (MC corrected)",
            **sig_kw,
            histtype="step",
            weights=bkg_wts * bkg_mc_wts,
            linestyle=":",
        )

        axis.set_title(col)

    axes[0, 0].legend()
    fig.tight_layout()

    path = "training_vars.svg"
    print(f"plotting {path}")
    fig.savefig(path)

    plt.close(fig)


def main(year: str, sign: str, magnetisation: str):
    """
    Create the classifier, print training scores

    """
    # If classifier already exists, tell us and exit
    clf_path = definitions.classifier_path(year, sign, magnetisation)
    if clf_path.is_file():
        print(f"{clf_path} exists")
        return

    classifier_dir = pathlib.Path(__file__).resolve().parents[0] / "classifiers"
    if not classifier_dir.is_dir():
        os.mkdir(str(classifier_dir))

    # Label 1 for signal; 0 for bkg
    (
        train_df,
        test_df,
        train_label,
        test_label,
        train_d0_wts,
        test_d0_wts,
    ) = _train_test_dfs(year, sign, magnetisation)

    # We want to train the classifier on a realistic proportion of signal + background
    # Get this from running `scripts/mass_fit.py`
    # using this number for now
    sig_frac = 0.0852
    train_weights = util.weights(train_label, sig_frac, train_d0_wts)

    # Plot delta M and D mass distributions before weighting
    _plot_masses(
        train_df, train_label, np.ones_like(train_label), "training_masses_no_wt.png"
    )

    # Plot delta M and D mass distributions
    _plot_masses(train_df, train_label, train_weights, "training_masses_weighted.png")
    _plot_masses(
        train_df,
        train_label,
        train_weights * train_d0_wts,
        "training_masses_d0_weighted.png",
    )

    # Plot the training data
    _plot_train_vars(train_df, train_label, train_weights, train_d0_wts)

    # Type is defined in lib_cuts.definitions
    clf = definitions.Classifier(
        n_estimators=100, max_depth=8, learning_rate=0.15, loss="exponential"
    )

    # We only want to use some of our variables for training
    training_labels = list(training_vars.training_var_names())
    clf.fit(train_df[training_labels], train_label, train_weights * train_d0_wts)

    print(classification_report(train_label, clf.predict(train_df[training_labels])))

    # Resample for the testing part of the classification report
    gen = np.random.default_rng()
    test_mask = util.resample_mask(gen, test_label, sig_frac, test_d0_wts)
    print(
        classification_report(
            test_label[test_mask], clf.predict(test_df[training_labels][test_mask])
        )
    )

    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    main(args.year, args.sign, args.magnetisation)
