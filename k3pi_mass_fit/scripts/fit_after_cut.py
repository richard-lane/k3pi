"""
Example script perhaps

"""
import os
import sys
import glob
import tqdm
import pickle
import uproot
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from hep_ml.uboost import uBoostBDT
from iminuit.cost import ExtendedUnbinnedNLL

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_data import get
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars
from libFit import fit


def _plot(dataframe: pd.DataFrame, path: str):
    """
    Plot histograms of every variable in the dataframe

    """
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    kw = {"histtype": "step", "density": True}
    for a, label in zip(ax.ravel(), dataframe):
        # Find histogram bins
        lo, hi = np.quantile(dataframe[label], [0.05, 0.95])
        bins = np.linspace(0.9 * lo, 1.1 * hi, 100)

        # Plot
        a.hist(dataframe[label], label="data", bins=bins, **kw)

        a.set_title(label)
        a.legend()

    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main():
    # Read data
    year, sign, magnetisation = "2018", "dcs", "magdown"
    dataframe = pd.concat(get.data(year, sign, magnetisation))

    clf = get_clf(year, sign, magnetisation)

    # TODO should really do this with a different threshhold
    training_labels = list(training_vars.training_var_names())
    predicted_signal = clf.predict(dataframe[training_labels]) == 1

    # Plot training vars
    _plot(dataframe, "data_vars_all.png")
    _plot(dataframe[predicted_signal], "data_vars_cut.png")

    # Do a mass fit to the thing we just removed background from with the BDT
    fitter = fit.fit(dataframe["Delta_M"][predicted_signal].to_numpy(), sign, 5, 0.8)

    # Should also probably plot the mass fit as well


if __name__ == "__main__":
    main()
