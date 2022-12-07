"""
Add phsp bin index column to existing dataframes

"""
import os
import sys
import glob
import pickle
import pathlib
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_data import definitions
from lib_efficiency.amplitude_models import amplitudes, definitions as ampdef
from lib_efficiency.efficiency_util import k_3pi


def _interference_term(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Find the interference (amplitudes cross term) from a dataframe

    returns mask of K+, the K+ terms, the K- terms

    """
    # Create a mask of K+, K- arrays
    k_plus = dataframe["K ID"] == 321

    k3pi = tuple(a.astype(np.float64) for a in k_3pi(dataframe))

    # Find the K- amplitudes
    k3pi_plus = [p[:, k_plus] for p in k3pi]
    plus_cf = amplitudes.cf_amplitudes(*k3pi_plus, +1)
    plus_dcs = amplitudes.dcs_amplitudes(*k3pi_plus, +1) * ampdef.DCS_OFFSET

    # Find the K- amplitudes
    k3pi_minus = [p[:, ~k_plus] for p in k3pi]
    minus_cf = amplitudes.cf_amplitudes(*k3pi_minus, -1)
    minus_dcs = amplitudes.dcs_amplitudes(*k3pi_minus, -1) * ampdef.DCS_OFFSET

    # Find the angle between them
    plus_interference = plus_dcs.conj() * plus_cf
    minus_interference = minus_dcs.conj() * minus_cf

    return k_plus, plus_interference, minus_interference


def _plot_hist(
    plus_term: np.ndarray,
    minus_term: np.ndarray,
    phsp_bins: Tuple,
    args: argparse.Namespace,
) -> None:
    """Plot a histogram of angles"""
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    fig, axis = plt.subplots(figsize=(10, 10))
    hist_kw = {"histtype": "step", "bins": np.linspace(-180.0, 180.0, 200)}

    axis.hist(plus_term, **hist_kw, label=r"$D\rightarrow K^+3\pi$")
    axis.hist(minus_term, **hist_kw, label=r"$D\rightarrow K^-3\pi$")

    for lim in phsp_bins:
        axis.axvline(lim, color="k")

    axis.set_xlabel(r"arg($A^{WS}\times A^{RS}$) / $\degree$")
    axis.set_title(f"{year} {sign} {magnetisation}")

    axis.legend()
    fig.savefig("phsp_bins.png")

    plt.show()


def _plot_scatter(
    plus_term: np.ndarray,
    minus_term: np.ndarray,
    phsp_bins: Tuple,
    args: argparse.Namespace,
) -> None:
    """Plot a histogram of angles"""
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    plus_angles = np.angle(plus_term)
    minus_angles = np.angle(minus_term)

    bins_rad = [x * np.pi / 180.0 for x in phsp_bins]

    plus_indices = np.digitize(plus_angles, bins_rad) - 1
    minus_indices = np.digitize(minus_angles, bins_rad) - 1

    plus_mag = np.abs(plus_term)
    minus_mag = np.abs(minus_term)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10), subplot_kw={"projection": "polar"})

    for index, _ in enumerate(phsp_bins):
        axes[0].scatter(
            plus_angles[plus_indices == index],
            plus_mag[plus_indices == index],
            label=f"Bin {index}",
            s=1,
        )
        axes[1].scatter(
            minus_angles[minus_indices == index],
            minus_mag[minus_indices == index],
            label=f"Bin {index}",
            s=1,
        )

    for lim in bins_rad:
        for axis in axes:
            axis.axvline(lim, color="k")

    fig.suptitle(f"{year} {sign} {magnetisation}")

    axes[0].set_title(r"$D\rightarrow K^+$")
    axes[1].set_title(r"$D\rightarrow K^-$")

    axes[1].legend()
    fig.savefig("phsp_bin_scatter.png")

    plt.show()


def main(args: argparse.Namespace) -> None:
    """Add phsp bin info to dataframes"""
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    dump_paths = glob.glob(str(definitions.data_dir(year, sign, magnetisation) / "*"))

    phsp_bins = (-180.0, -39.0, 0.0, 43.0, 180.0)

    # List of arrays
    # Hopefully we won't run out of memory
    plus_term = []
    minus_term = []

    col_header = "phsp bin"

    for path in tqdm(dump_paths):
        if os.path.isdir(path):
            continue

        # Open the dataframe
        with open(path, "rb") as f:
            dataframe = pickle.load(f)

        if col_header in dataframe:
            continue

        # Find K+ and K- type interference terms
        k_plus, plus_interference, minus_interference = _interference_term(dataframe)

        # Find the corresponding angle
        plus_angles = np.angle(plus_interference, deg=True)
        minus_angles = np.angle(minus_interference, deg=True)

        # Bin these angles in terms of phsp bins
        plus_indices = np.digitize(plus_angles, phsp_bins) - 1
        minus_indices = np.digitize(minus_angles, phsp_bins) - 1

        # Create an empty array of bin numbers
        bin_indices = np.ones(len(dataframe), dtype=int) * np.nan
        bin_indices[k_plus] = plus_indices
        bin_indices[~k_plus] = minus_indices

        # Add this column to the dataframe
        dataframe[col_header] = bin_indices

        # Dump it
        with open(path, "wb") as f:
            pickle.dump(dataframe, f)

        # Append to list of arrays
        plus_term.append(plus_interference)
        minus_term.append(minus_interference)

    # Plot a histogram
    plus_angles = np.concatenate([np.angle(x, deg=True) for x in plus_term])
    minus_angles = np.concatenate([np.angle(x, deg=True) for x in minus_term])

    _plot_hist(plus_angles, minus_angles, phsp_bins, args)

    plus_term = np.concatenate(plus_term)
    minus_term = np.concatenate(minus_term)
    _plot_scatter(plus_term, minus_term, phsp_bins, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a column for phase space bin index to the DataFrames to `dumps/`"
    )
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

    main(parser.parse_args())
