"""
Visualise the phase space bins

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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import phsp_binning, get


def _plot_hist(
    interference_terms: List[np.ndarray], dataframes: Iterable
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of angles"""
    # Find angles
    plus_angles = [[] for _ in range(5)]
    minus_angles = [[] for _ in range(5)]

    for term, dataframe in tqdm(zip(interference_terms, dataframes)):
        k_plus = dataframe["K ID"] == 321
        index = dataframe["phsp bin"]

        angles = np.angle(term, deg=True)
        for i in range(5):
            plus_angles[i].append(angles[k_plus][index[k_plus] == i])
            minus_angles[i].append(angles[~k_plus][index[~k_plus] == i])

    plus_angles = [np.concatenate(arr) for arr in plus_angles]
    minus_angles = [np.concatenate(arr) for arr in minus_angles]

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    hist_kw = {"histtype": "step", "bins": np.arange(361) - 180.0}

    for plus, minus in zip(plus_angles, minus_angles):
        axes[0].hist(plus, **hist_kw)
        axes[1].hist(minus, **hist_kw)

    axes[0].set_title(r"$D\rightarrow K^+3\pi$")
    axes[1].set_title(r"$D\rightarrow K^-3\pi$")

    for axis in axes:
        for lim in phsp_binning.BINS:
            axis.axvline(lim, color="k")

        axis.set_xlabel(r"arg($A^{WS}\times A^{RS}$) / $\degree$")

    return fig, axes


def _plot_scatter(
    interference_terms: List[np.ndarray], dataframes: Iterable[pd.DataFrame]
) -> None:
    """Plot a histogram of angles"""
    bins_rad = [x * np.pi / 180.0 for x in phsp_binning.BINS]

    plus_angles = [[] for _ in range(5)]
    minus_angles = [[] for _ in range(5)]

    plus_mag = [[] for _ in range(5)]
    minus_mag = [[] for _ in range(5)]
    for term, dataframe in tqdm(zip(interference_terms, dataframes)):
        k_plus = dataframe["K ID"] == 321
        angles = np.angle(term)
        index = dataframe["phsp bin"]

        for i in range(5):
            plus_angles[i].append(angles[k_plus][index[k_plus] == i])
            minus_angles[i].append(angles[~k_plus][index[~k_plus] == i])

            plus_mag[i].append(np.abs(term[k_plus][index[k_plus] == i]))
            minus_mag[i].append(np.abs(term[~k_plus][index[~k_plus] == i]))

    plus_angles = [np.concatenate(arr) for arr in plus_angles]
    minus_angles = [np.concatenate(arr) for arr in minus_angles]

    plus_mag = [np.concatenate(arr) for arr in plus_mag]
    minus_mag = [np.concatenate(arr) for arr in minus_mag]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"})

    colours = "r", "g", "b", "y", "k"
    alphas = (1.0, 1.0, 1.0, 1.0, 0.3)
    for i, (angle, mag, colour, alpha) in enumerate(
        zip(plus_angles, plus_mag, colours, alphas)
    ):
        axes[0].scatter(angle, mag, label=f"Bin {i}", s=0.5, color=colour, alpha=alpha)

    for i, (angle, mag, colour, alpha) in enumerate(
        zip(minus_angles, minus_mag, colours, alphas)
    ):
        axes[1].scatter(angle, mag, label=f"Bin {i}", s=0.5, color=colour, alpha=alpha)

    for axis in axes:
        for lim in bins_rad:
            axis.axvline(lim, color="k", linestyle="--")

    axes[0].set_title(r"$D\rightarrow K^+$")
    axes[1].set_title(r"$D\rightarrow K^-$")

    axes[1].legend()

    return fig, axes


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
    fig, axes = _plot_hist(
        (
            phsp_binning.interference_terms(dataframe)
            for dataframe in _dataframes(year, sign, magnetisation)
        ),
        _dataframes(year, sign, magnetisation),
    )
    fig.tight_layout()

    suffix = f"_{year}_{sign}_{magnetisation}.png"
    fig.savefig(f"phsp_bin_hist{suffix}")
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
    fig.savefig(f"phsp_bin_scatter{suffix}")


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
