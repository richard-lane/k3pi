"""
Plot real data variables before and after cuts

"""
import sys
import pickle
import pathlib
import argparse
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_cuts.definitions import THRESHOLD
from lib_data import get, training_vars, cuts


def _plot(
    axis: plt.Axes,
    var: np.ndarray,
    predictions: np.ndarray,
    bins: np.ndarray,
) -> None:
    """
    Plot variables on an axis before/after cuts

    """
    axis.hist(var, bins=bins, label="before", histtype="step", color="b")
    axis.hist(var[predictions == 1], bins=bins, label="after", alpha=0.5, color="b")


def main(*, year: str, sign: str, magnetisation: str):
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    n_dfs = 150
    dataframes = list(
        tqdm(
            islice(cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation)), n_dfs),
            total=n_dfs,
        )
    )
    dataframe = pd.concat(dataframes)

    # Predict which of these are signal and background using our classifier
    # trained the clf on DCS
    clf = get_clf(year, "dcs", magnetisation)

    training_labels = list(training_vars.training_var_names())

    threshhold = THRESHOLD
    predictions = clf.predict_proba(dataframe[training_labels])[:, 1] > threshhold
    print(f"{threshhold=}\tefficiency: {np.sum(predictions) / len(predictions)}")

    # The histogram bin limits
    n_bins = 100
    bins = [
        np.linspace(0, 75, n_bins),
        np.linspace(0, 40, n_bins),
        np.linspace(0, 10, n_bins),
        np.linspace(0, 1, n_bins),
        np.linspace(0, 1500, n_bins),
        np.linspace(0, 30, n_bins),
        np.linspace(1840, 1900, n_bins),
        np.linspace(1980, 2050, n_bins),
        np.linspace(139, 152, n_bins),
    ]
    units = ["", "", "", "", "MeV", "", "MeV", "MeV", "MeV"]
    # Plot histograms of our variables before/after doing these cuts
    columns = list(training_vars.training_var_names()) + ["Dst_ReFit_D0_M", "D* mass"]
    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    for col, axis, bin_, unit in zip(columns, ax.ravel(), bins[:-1], units[:-1]):
        _plot(axis, dataframe[col], predictions, bin_)

        axis.set_title(col if col in training_vars.training_var_names() else col + "*")
        axis.set_xlabel(unit)

    # Let's also plot the mass difference
    _plot(
        ax.ravel()[-1],
        dataframe["D* mass"] - dataframe["Dst_ReFit_D0_M"],
        predictions,
        bins[-1],
    )
    ax.ravel()[-1].set_title(r"$\Delta$M*")
    ax.ravel()[-1].set_xlabel(units[-1])

    ax[0, 0].legend()

    # fig.suptitle(
    #     f"Data before/after BDT cut; {year} {sign} {magnetisation}\n {threshhold=}"
    # )

    fig.tight_layout()

    path = f"{sign}_data_cuts.png"
    plt.savefig(path)

    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, ax), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot effect of BDT cut on training data"
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
