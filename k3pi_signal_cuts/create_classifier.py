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
from sklearn.metrics import classification_report

from lib_cuts import definitions
from lib_cuts import util

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "k3pi-data"))

from lib_data import get, training_vars


def _train_test_dfs(year, sign, magnetisation):
    """ Join signal + bkg dataframes but split by train/test """
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    combined_df = pd.concat((sig_df, bkg_df))

    # 1 for signal, 0 for background
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    train_mask = combined_df["train"]
    return (
        combined_df[train_mask],
        combined_df[~train_mask],
        labels[train_mask],
        labels[~train_mask],
    )


def _plot_masses(
    train_df: pd.DataFrame, train_label: np.ndarray, weights: np.ndarray, path: str
) -> None:
    """
    Saves to train_masses.png

    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    bkg_df = train_df[train_label == 0]
    sig_df = train_df[train_label == 1]

    bkg_wt = weights[train_label == 0]
    sig_wt = weights[train_label == 1]

    def _delta_m(dataframe):
        return dataframe["D* mass"] - dataframe["D0 mass"]

    ax[0].hist(
        sig_df["D0 mass"],
        bins=np.linspace(1775, 1950, 200),
        color="b",
        alpha=0.4,
        weights=sig_wt,
    )
    ax[0].hist(
        bkg_df["D0 mass"],
        bins=np.linspace(1775, 1950, 200),
        color="r",
        alpha=0.4,
        weights=bkg_wt,
    )

    ax[1].hist(
        _delta_m(sig_df),
        bins=np.linspace(140, 160, 200),
        color="b",
        alpha=0.4,
        label="sig",
        weights=sig_wt,
    )
    ax[1].hist(
        _delta_m(bkg_df),
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
    train_df, test_df, train_label, test_label = _train_test_dfs(
        year, sign, magnetisation
    )

    # We want to train the classifier on a realistic proportion of signal + background
    # Get this from running `scripts/mass_fit.py`
    # using this number for now
    sig_frac = 0.0852
    train_weights = util.weights(train_label, sig_frac)

    # Plot delta M and D mass distributions before weighting
    _plot_masses(
        train_df, train_label, np.ones_like(train_label), "training_masses_no_wt.png"
    )

    # Plot delta M and D mass distributions
    _plot_masses(train_df, train_label, train_weights, "training_masses_weighted.png")

    # Type is defined in lib_cuts.definitions
    clf = definitions.Classifier()

    # We only want to use some of our variables for training
    training_labels = list(training_vars.training_var_names())
    clf.fit(train_df[training_labels], train_label, train_weights)

    print(classification_report(train_label, clf.predict(train_df[training_labels])))

    # Resample for the testing part of the classification report
    gen = np.random.default_rng()
    test_mask = util.resample_mask(gen, test_label, sig_frac)
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
        choices={"2018"},
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
        choices={"magdown"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation)
