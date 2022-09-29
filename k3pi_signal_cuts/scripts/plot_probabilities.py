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

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = get_clf(year, sign, magnetisation)
    var_names = list(training_vars.training_var_names())
    sig_proba = clf.predict_proba(sig_df[var_names])[:, 1]
    bkg_proba = clf.predict_proba(bkg_df[var_names])[:, 1]

    # Plot histograms of our variables before/after doing these cuts
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 100)
    ax.hist(sig_proba, bins=bins, label="sig")
    ax.hist(bkg_proba, bins=bins, label="bkg")

    ax.legend()
    fig.suptitle("signal probability")

    plt.show()


if __name__ == "__main__":
    main()
