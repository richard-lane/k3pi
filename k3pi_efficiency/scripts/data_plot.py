"""
Show the effect of the efficiency weighting on real data

"""
import sys
import pathlib
from typing import Iterable
from multiprocessing import Process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

import common
from lib_data import get, stats
from lib_data.util import k_3pi
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_efficiency.efficiency_util import points
from libFit.util import delta_m_generator
from libFit.definitions import mass_bins


def _dataframes(
    year: str, decay_type: str, magnetisation: str
) -> Iterable[pd.DataFrame]:
    """
    Get a generator of the right dataframes

    """
    return get.data(year, decay_type, magnetisation)


def _weights(
    year: str, decay_type: str, magnetisation: str, reweighter: EfficiencyWeighter
) -> Iterable[np.ndarray]:
    """
    Efficiency weights

    """
    return (
        reweighter.weights(points(*k_3pi(dataframe), dataframe["time"]))
        for dataframe in _dataframes(year, decay_type, magnetisation)
    )


def _delta_m(year: str, decay_type: str, magnetisation: str) -> Iterable[np.ndarray]:
    """
    Generator of delta M

    """
    return delta_m_generator(_dataframes(year, decay_type, magnetisation))


def _plot_delta_m(
    year: str, decay_type: str, magnetisation: str, reweighter: EfficiencyWeighter
) -> None:
    """
    Make a plot of delta M before and after the correction

    """
    bins = mass_bins(200)

    count_before, err_before = stats.counts_generator(
        _delta_m(year, decay_type, magnetisation), bins
    )

    count_after, err_after = stats.counts_generator(
        _delta_m(year, decay_type, magnetisation),
        bins,
        _weights(year, decay_type, magnetisation, reweighter),
    )

    centres = (bins[1:] + bins[:-1]) / 2.0
    half_widths = (bins[1:] - bins[:-1]) / 2.0
    fig, axis = plt.subplot_mosaic("AAA\n" * 5 + "BBB", figsize=(8, 10), sharex=True)

    # Plot the histograms
    errorbar_kw = {
        "x": centres,
        "xerr": half_widths,
        "markersize": 0.1,
        "elinewidth": 0.5,
    }
    axis["A"].errorbar(
        y=count_before,
        yerr=err_before,
        fmt="k.",
        label="Raw",
        **errorbar_kw,
    )
    axis["A"].errorbar(
        y=count_after, yerr=err_after, fmt="b.", label="Corrected", **errorbar_kw
    )

    # Plot the difference
    diff = count_after - count_before
    diff_err = np.sqrt(err_after**2 + err_before**2)

    axis["B"].axhline(0.0, color="k")
    axis["B"].errorbar(y=diff / diff_err, yerr=1.0, fmt="k.", **errorbar_kw)

    # Legends, labels, etc
    axis["A"].legend()

    axis["A"].set_ylabel("count")

    axis["B"].set_xlabel(r"$\Delta M$")
    axis["B"].set_ylabel(r"$\Delta$ count (normalised)")

    fig.suptitle(
        f"{year} {decay_type} {magnetisation} Efficiency correction\n Time, phsp integrated"
    )

    fig.tight_layout()

    path = f"eff_delta_m_{year}_{decay_type}_{magnetisation}.png"
    print(f"saving {path}")
    fig.savefig(path)


def _phsp_points(
    year: str, decay_type: str, magnetisation: str
) -> Iterable[np.ndarray]:
    """
    Generator of (one dimension of) phase space points

    """
    return (
        points(*k_3pi(dataframe), dataframe["time"])[:, -2]
        for dataframe in _dataframes(year, decay_type, magnetisation)
    )


def _plot_phsp(
    year: str, decay_type: str, magnetisation: str, reweighter: EfficiencyWeighter
) -> None:
    """
    Make a plot of phase space parameters before and after the correction

    """
    # Only plot one of the dimensions
    bins = np.linspace(-np.pi, np.pi, 201)

    count_before, err_before = stats.counts_generator(
        _phsp_points(year, decay_type, magnetisation), bins
    )

    count_after, err_after = stats.counts_generator(
        _phsp_points(year, decay_type, magnetisation),
        bins,
        _weights(year, decay_type, magnetisation, reweighter),
    )

    centres = (bins[1:] + bins[:-1]) / 2.0
    half_widths = (bins[1:] - bins[:-1]) / 2.0
    fig, axis = plt.subplot_mosaic("AAA\n" * 5 + "BBB", figsize=(8, 10), sharex=True)

    # Plot the histograms
    errorbar_kw = {
        "x": centres,
        "xerr": half_widths,
        "markersize": 0.1,
        "elinewidth": 0.5,
    }
    axis["A"].errorbar(
        y=count_before,
        yerr=err_before,
        fmt="k.",
        label="Raw",
        **errorbar_kw,
    )
    axis["A"].errorbar(
        y=count_after, yerr=err_after, fmt="b.", label="Corrected", **errorbar_kw
    )

    # Plot the difference
    diff = count_after - count_before
    diff_err = np.sqrt(err_after**2 + err_before**2)

    axis["B"].axhline(0.0, color="k")
    axis["B"].errorbar(y=diff / diff_err, yerr=1.0, fmt="k.", **errorbar_kw)

    # Legends, labels, etc
    axis["A"].legend()

    axis["A"].set_ylabel("count")

    axis["B"].set_xlabel(r"$\Delta M$")
    axis["B"].set_ylabel(r"$\Delta$ count (normalised)")

    fig.suptitle(
        f"{year} {decay_type} {magnetisation} Efficiency correction\n Time, phsp integrated"
    )

    fig.tight_layout()

    path = f"eff_phi_{year}_{decay_type}_{magnetisation}.png"
    print(f"saving {path}")
    fig.savefig(path)


def main(
    *,
    year: str,
    decay_type: str,
    weighter_type: str,
    magnetisation: str,
    data_k_charge: str,
    weighter_k_charge: str,
    fit: bool,
    cut: bool,
):
    """
    Make plots of delta M and the phase space

    """
    if decay_type not in {"dcs", "cf"}:
        raise ValueError("cant have false sign data")

    # Generator of real data dataframes
    if cut:
        raise NotImplementedError("bdt cut the dfs")

    if data_k_charge != "both":
        raise NotImplementedError("haven't done this yet")

    # Open reweighter
    reweighter = get_reweighter(
        year,
        weighter_type,
        magnetisation,
        weighter_k_charge,
        fit=fit,
        cut=cut,
        verbose=True,
    )

    # Delta M plot
    procs = [
        Process(target=fcn, args=(year, decay_type, magnetisation, reweighter))
        for fcn in (_plot_delta_m, _plot_phsp)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # Time plot


if __name__ == "__main__":
    parser = common.parser("Plot projections of phase space variables for real data")
    main(**vars(parser.parse_args()))
