"""
Visualise the phase space bins, for the plot in thesis

Unlike the script in k3pi-data/scripts, does not split the data
by Kaon charge

Must have added columns with the `k3pi-data/scripts/add_phsp_bins.py`
script first

"""
import sys
import pathlib
import argparse
from typing import Tuple, List, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import phsp_binning, get


def _plot_hist(
    interference_terms: List[np.ndarray], dataframes: Iterable
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of angles"""
    # Find angles
    angles = [[] for _ in range(5)]

    for term, dataframe in tqdm(zip(interference_terms, dataframes)):
        index = dataframe["phsp bin"]

        tmp_angles = np.angle(term, deg=True)
        for i in range(5):
            angles[i].append(tmp_angles[index == i])

    angles = [np.concatenate(arr) for arr in angles]

    fig, axis = plt.subplots(1, 1, figsize=(10, 10))
    hist_kw = {"histtype": "step", "bins": np.arange(361) - 180.0}

    colours = "r", "g", "b", "y", "k"
    axis.hist(angles, **hist_kw, color=colours)

    ticks = [-180.0, -90.0, 0.0, 90.0, 180.0]
    for lim in phsp_binning.BINS:
        axis.axvline(lim, color="k", linestyle="--")

    axis.set_xlabel(r"arg($A^{\mathrm{dcs}}\times A^{\mathrm{cf}}$) / $\degree$")
    axis.set_xticks(ticks)

    return fig, axis


def _plot_scatter(
    interference_terms: List[np.ndarray], dataframes: Iterable[pd.DataFrame]
) -> None:
    """Plot a histogram of angles"""
    bins_rad = [x * np.pi / 180.0 for x in phsp_binning.BINS]

    angles = [[] for _ in range(5)]
    mag = [[] for _ in range(5)]

    for term, dataframe in tqdm(zip(interference_terms, dataframes)):
        tmp_angles = np.angle(term)
        index = dataframe["phsp bin"]

        for i in range(5):
            angles[i].append(tmp_angles[index == i])
            mag[i].append(np.abs(term[index == i]))

    angles = [np.concatenate(arr) for arr in angles]
    mag = [np.concatenate(arr) for arr in mag]

    fig, axis = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "polar"})

    colours = "r", "g", "b", "y", "k"
    alphas = (1.0, 1.0, 1.0, 1.0, 0.3)
    for i, (angle, mag, colour, alpha) in enumerate(zip(angles, mag, colours, alphas)):
        axis.scatter(angle, mag, label=f"Bin {i}", s=0.5, color=colour, alpha=alpha)

    for lim in bins_rad:
        axis.axvline(lim, color="k", linestyle="--")

    axis.legend()

    return fig, axis


def _dataframes(year: str, sign: str, magnetisation: str) -> Iterable[pd.DataFrame]:
    """
    Iterable of dataframes

    """
    return get.data(year, sign, magnetisation)


def main(year: str, sign: str, magnetisation: str):
    """
    Plot a histogram of the interference term
    angle, and a scatter plot in each bin

    """
    # Plot the interference terms as a histogram
    fig, _ = _plot_hist(
        (
            phsp_binning.interference_terms(dataframe)
            for dataframe in _dataframes(year, sign, magnetisation)
        ),
        _dataframes(year, sign, magnetisation),
    )
    fig.tight_layout()

    suffix = f"_{year}_{sign}_{magnetisation}.png"
    path = str(pathlib.Path(__file__).resolve().parents[0] / f"phsp_bin_hist{suffix}")
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)

    # Plot the interference terms as a scatter plot
    fig, _ = _plot_scatter(
        (
            phsp_binning.interference_terms(dataframe)
            for dataframe in _dataframes(year, sign, magnetisation)
        ),
        _dataframes(year, sign, magnetisation),
    )
    fig.tight_layout()
    path = str(
        pathlib.Path(__file__).resolve().parents[0] / f"phsp_bin_scatter{suffix}"
    )
    print(f"saving {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot phase space binning")
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

    main(**vars(parser.parse_args()))
