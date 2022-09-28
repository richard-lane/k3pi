"""
Plot real data variables before and after cuts

"""
import sys
import pathlib
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def _plot(
    axis: plt.Axes,
    var: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """
    Plot variables on an axis before/after cuts

    """
    # 100 bins between the 1% and 99% quantiles
    bins = np.linspace(*np.quantile(var, [0.01, 0.99]), 100)

    axis.hist(var, bins=bins, label="before", histtype="step", color="b")
    axis.hist(var[predictions == 1], bins=bins, label="after", alpha=0.5, color="b")


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    year, sign, magnetisation = "2018", "cf", "magdown"

    n_dfs = 50
    dataframes = list(
        tqdm(islice(get.data(year, sign, magnetisation), n_dfs), total=n_dfs)
    )
    dataframe = pd.concat(dataframes)

    # Predict which of these are signal and background using our classifier
    # trained the clf on DCS, i think...
    clf = get_clf(year, "dcs", magnetisation)

    training_labels = list(training_vars.training_var_names())

    threshhold = 0.180
    predictions = clf.predict_proba(dataframe[training_labels])[:, 1] > threshhold

    # Plot histograms of our variables before/after doing these cuts
    columns = list(training_vars.training_var_names()) + ["D0 mass", "D* mass"]
    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for col, axis in zip(columns, ax.ravel()):
        _plot(axis, dataframe[col], predictions)

        axis.set_title(col if col in training_vars.training_var_names() else col + "*")

    # Let's also plot the mass difference
    _plot(
        ax.ravel()[-1],
        dataframe["D* mass"] - dataframe["D0 mass"],
        predictions,
    )
    ax.ravel()[-1].set_title(r"$\Delta$M*")

    ax[0, 0].legend()

    fig.suptitle(f"Data before/after BDT cut; {year} {sign} {magnetisation}\n {threshhold=}")

    fig.tight_layout()

    plt.savefig("data_cuts.png")

    plt.show()


if __name__ == "__main__":
    main()
