"""
Histograms of classification probabilities for testing data

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = util.get_classifier()
    sig_proba = clf.predict_proba(sig_df[list(read_data.training_var_names())])[:, 1]
    bkg_proba = clf.predict_proba(bkg_df[list(read_data.training_var_names())])[:, 1]

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
