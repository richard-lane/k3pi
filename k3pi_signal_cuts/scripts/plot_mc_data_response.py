"""
Histograms of classification probabilities for MC and real data,
to check that the MC looks like the data

"""
import sys
import pickle
import pathlib
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_cuts.definitions import THRESHOLD
from lib_data import get, training_vars


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    # Want CF as CF data is mostly signal
    # We can therefore compare it to MC
    fig, ax = plt.subplots()

    year, magnetisation = "2018", "magdown"
    clf = get_clf(year, "dcs", magnetisation)

    # Trained the classifer on DCS stuff
    sig_df = get.mc(year, "dcs", magnetisation)
    bkg_df = pd.concat(get.uppermass(year, "dcs", magnetisation))

    # Compare it to RS data
    data_dfs = islice(get.data(year, "cf", magnetisation), None)

    # Predict which of these are signal and background using our classifier
    var_names = list(training_vars.training_var_names())

    sig_proba = clf.predict_proba(sig_df[var_names])[:, 1]
    bkg_proba = clf.predict_proba(bkg_df[var_names])[:, 1]
    data_probas = (clf.predict_proba(data_df[var_names])[:, 1] for data_df in data_dfs)

    # Plot histograms of probability
    bins = np.linspace(0, 1, 100)
    hist_kw = {"bins": bins, "histtype": "step", "density": True}
    ax.hist(sig_proba, label="Signal (CF MC)", **hist_kw)
    ax.hist(bkg_proba, label="Bkg (Upper Mass Sideband)", **hist_kw)
    ax.hist(np.concatenate(list(data_probas)), label="RS Data", **hist_kw)

    # Plot the WS data as well
    # data_dfs = islice(get.data(year, "dcs", magnetisation), 50)
    # data_probas = (clf.predict_proba(data_df[var_names])[:, 0] for data_df in data_dfs)
    # ax.hist(np.concatenate(list(data_probas)), label="WS Data", **hist_kw)

    ax.legend()
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count (normalised)")
    ax.set_title("BDT Signal Classification Probability")

    ax.axvline(THRESHOLD, color="r")

    fig.tight_layout()

    plt.savefig("mc_data_response.png")
    with open("plot_pkls/mc_data_response.pkl", "wb") as f:
        pickle.dump((fig, ax), f)


if __name__ == "__main__":
    main()
