"""
See if we can predict delta M from D0 pT and slow pi pT

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classifier
from sklearn.metrics import classification_report

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
from lib_data import get


def _arrays(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns array of (D0 pT, pi pT) and delta M bin indices

    """
    # Bin delta M
    delta_m = dataframe["D* mass"] - dataframe["D0 mass"]

    return (
        np.column_stack((dataframe[r"D0 $p_T$"], dataframe[r"$\pi_s$ $p_T$"])),
        delta_m,
    )


def _classify(dataframe: pd.DataFrame, path: str) -> None:
    """
    Train a classifier to predict which delta M bin an event is in

    Plot what proportion of our guesses are correct for each bin, i.e. the precision

    Also then print classification report

    """
    # Get the columns we want
    p_t, delta_m = _arrays(dataframe)

    # Bin delta M
    n_bins = 7
    bins = np.quantile(delta_m, [i / n_bins for i in range(n_bins + 1)])

    bins[
        -1
    ] *= 1.0001  # Make the last bin a bit bigger to include all the points nicely
    indices = np.digitize(delta_m, bins) - 1

    # Train some sort of classifier
    train = np.random.random(len(p_t)) < 0.5
    clf = Classifier(n_neighbors=150, weights="uniform", n_jobs=4)
    clf.fit(p_t[train], indices[train])

    predictions = clf.predict(p_t[~train])
    correct = predictions == indices[~train]

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    frac_correct = []
    err = []

    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True)
    hist_bins = np.linspace(bins[0], bins[-1], 200)
    for i in np.unique(indices):
        # Compare how many we get right in this bin to how many we guess are in this bin overall
        true_in_this_bin = indices[~train] == i
        num_correct = np.sum(correct[true_in_this_bin])
        guessed_in_this_bin = np.sum(predictions == i)

        frac_correct.append(100 * num_correct / guessed_in_this_bin)
        err.append(
            frac_correct[-1] * np.sqrt(1 / num_correct + 1 / guessed_in_this_bin)
        )

        ax["B"].hist(delta_m[~train][true_in_this_bin], bins=hist_bins)

    ax["A"].errorbar(centres, frac_correct, xerr=widths, yerr=err, fmt="k.")
    ax["A"].axhline(100 / n_bins, color="r")
    ax["A"].text(plt.gca().get_xlim()[0] + 0.3, 100 / n_bins + 0.1, "Random Chance")
    ax["A"].set_ylabel("% predictions correct")
    ax["B"].set_xlabel(r"$\Delta$M")
    fig.savefig(path)

    print(
        "\t", classification_report(indices[~train], predictions).replace("\n", "\n\t")
    )


def _predict_deltam(sign: str):
    """
    For MC, upper mass sideband and real data:
        Train a classifier, find how good we are at predicting
        deltaM from the pTs

    """
    print("mc:")
    mc = get.mc("2018", sign, "magdown")
    _classify(mc, f"mc_{sign}_predict_deltam.png")

    print("upper mass:")
    uppermass = pd.concat(get.uppermass("2018", sign, "magdown"))
    _classify(uppermass, f"bkg_{sign}_predict_deltam.png")

    print("data:")
    data = pd.concat(get.data("2018", sign, "magdown"))
    _classify(data, f"data_{sign}_predict_deltam.png")


def main():
    """
    See if we can predict the value of delta M from the D0 and slow pi pT

    """
    _predict_deltam("dcs")
    _predict_deltam("cf")


if __name__ == "__main__":
    main()
