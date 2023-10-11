"""
Histograms of classification probabilities for testing data

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


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    sig_train_df = sig_df[sig_df["train"]]
    bkg_train_df = bkg_df[bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = get_clf(year, sign, magnetisation)
    var_names = list(training_vars.training_var_names())

    sig_train_proba = clf.predict_proba(sig_train_df[var_names])[:, 1]
    bkg_train_proba = clf.predict_proba(bkg_train_df[var_names])[:, 1]

    # Plot histograms of our variables before/after doing these cuts
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 100)
    hist_kw = {"histtype": "step", "bins": bins, "density": True}

    ax.hist(sig_train_proba, **hist_kw, label="Signal (Train)", color="k")
    ax.hist(bkg_train_proba, **hist_kw, label="Background (Train)", color="r")

    # Repeat with testing data
    sig_test_df = sig_df[~sig_df["train"]]
    bkg_test_df = bkg_df[~bkg_df["train"]]

    sig_test_proba = clf.predict_proba(sig_test_df[var_names])[:, 1]
    bkg_test_proba = clf.predict_proba(bkg_test_df[var_names])[:, 1]

    hist_kw["linestyle"] = "--"
    ax.hist(sig_test_proba, **hist_kw, label="Signal (Test)", color="k")
    ax.hist(bkg_test_proba, **hist_kw, label="Background (Test)", color="r")

    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Counts")

    fig.tight_layout()
    fig.savefig("clf_probabilities.png")

    plt.show()


if __name__ == "__main__":
    main()
