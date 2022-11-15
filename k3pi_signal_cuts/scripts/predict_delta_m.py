"""
See if we can predict delta M from D0 pT and slow pi pT

"""
import sys
import pathlib
from typing import Iterable, Tuple
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classifier
from sklearn.metrics import classification_report

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
from lib_data import get


def _arrays(
    dataframe: pd.DataFrame, column_names: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns array of training variables and delta M bin indices

    """
    delta_m = dataframe["D* mass"] - dataframe["D0 mass"]

    return (
        np.column_stack([dataframe[name] for name in column_names]),
        delta_m,
    )


def _classify(dataframe: pd.DataFrame, column_names: Iterable[str], path: str) -> None:
    """
    Train a classifier to predict which delta M bin an event is in

    Plot what proportion of our guesses are correct for each bin, i.e. the precision

    Also then print classification report

    """
    # Get the columns we want
    print(column_names)
    points, delta_m = _arrays(dataframe, column_names)

    # Bin delta M
    n_bins = 7
    bins = np.quantile(delta_m, [i / n_bins for i in range(n_bins + 1)])

    bins[
        -1
    ] *= 1.0001  # Make the last bin a bit bigger to include all the points nicely
    indices = np.digitize(delta_m, bins) - 1

    # Train some sort of classifier
    train = np.random.default_rng(seed=0).random(len(points)) < 0.5
    clf = Classifier(n_neighbors=150, weights="uniform", n_jobs=4)
    clf.fit(points[train], indices[train])

    predictions = clf.predict(points[~train])
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
    ax["A"].set_ylim(10, 35)
    fig.savefig(path)

    print(
        "\t", classification_report(indices[~train], predictions).replace("\n", "\n\t")
    )


def main():
    """
    See if we can predict the value of delta M from some of the BDT training variables

    """
    data = pd.concat(islice(get.data("2018", "dcs", "magdown"), 0, 16))

    _classify(data, (r"D0 $p_T$", r"$\pi_s$ $p_T$"), "predict_deltam_d0_pis_pT.png")
    _classify(data, (r"$\pi_s$ $p_T$",), "predict_deltam_pis_pT.png")
    _classify(data, (r"D0 $p_T$",), "predict_deltam_d0_pT.png")
    _classify(
        data,
        (r"D0 $p_T$", r"$K3\pi$ max $p_T$", r"$K3\pi$ min $p_T$", r"$K3\pi$ sum $p_T$"),
        "predict_deltam_pis_daughter_pT.png",
    )
    _classify(
        data,
        (
            r"D0 $p_T$",
            r"$\pi_s$ $p_T$",
            r"$K3\pi$ max $p_T$",
            r"$K3\pi$ min $p_T$",
            r"$K3\pi$ sum $p_T$",
        ),
        "predict_deltam_d0_pis_daughter_pT.png",
    )


if __name__ == "__main__":
    main()
