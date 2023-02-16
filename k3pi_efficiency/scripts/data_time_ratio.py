"""
Plot the ratio of DCS to CF times before/after the efficiency correction
(without the mass fit)

"""
import sys
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

import common
from lib_data import get, stats
from lib_data.util import k_3pi
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_efficiency.efficiency_util import points
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_time_fit.util import ratio_err


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


def _times(year: str, decay_type: str, magnetisation: str) -> Iterable[np.ndarray]:
    """
    Generator of decay times

    """
    return (
        dataframe["time"] for dataframe in _dataframes(year, decay_type, magnetisation)
    )


def _ratio_err(
    bins: np.ndarray,
    year: str,
    magnetisation: str,
    *,
    dcs_reweighter: EfficiencyWeighter = None,
    cf_reweighter: EfficiencyWeighter = None,
    dcs_sum_before: float = None,
    cf_sum_before: float = None,
):
    """
    Ratio and its error

    """
    dcs_wt = (
        None
        if dcs_reweighter is None
        else _weights(year, "dcs", magnetisation, dcs_reweighter)
    )
    cf_wt = (
        None
        if cf_reweighter is None
        else _weights(year, "cf", magnetisation, cf_reweighter)
    )

    dcs_count, dcs_err = stats.counts_generator(
        _times(year, "dcs", magnetisation), bins, dcs_wt
    )
    cf_count, cf_err = stats.counts_generator(
        _times(year, "cf", magnetisation), bins, cf_wt
    )

    # Discard over and underflow
    dcs_count = dcs_count[1:-1]
    cf_count = cf_count[1:-1]
    dcs_err = dcs_err[1:-1]
    cf_err = cf_err[1:-1]

    # If we're doing the reweighting, scale such that the weights are right
    # TODO get these properly from pgun dataframes and pgun_n_generated fcn
    if dcs_sum_before is not None:
        dcs_sum_after = np.sum(dcs_count)
        cf_sum_after = np.sum(cf_count)

        dcs_avg_eff, cf_avg_eff = 0.02033, 0.02064
        # I think this is wrong TODO
        dcs_scale = (dcs_sum_before / dcs_avg_eff) / (dcs_sum_after / dcs_sum_before)
        cf_scale = (cf_sum_before / cf_avg_eff) / (cf_sum_after / cf_sum_before)

        # I know this is wrong TODO
        dcs_scale = 1 / dcs_avg_eff
        cf_scale = 1 / cf_avg_eff

        print(f"{dcs_scale=}\n{cf_scale=}")

        dcs_count *= dcs_scale
        dcs_err *= dcs_scale

        cf_count *= cf_scale
        cf_err *= cf_scale

    return dcs_count, dcs_err, cf_count, cf_err


def _bins():
    """
    Get time bins from CF time quantiles

    """
    n_bins = 100

    return np.concatenate(
        (
            [-np.inf],
            np.exp(np.linspace(np.log(MIN_TIME), np.log(10.0), n_bins)),
            [np.inf],
        )
    )


def _plot_time_ratio(
    year: str,
    magnetisation: str,
    cf_reweighter: EfficiencyWeighter,
    dcs_reweighter: EfficiencyWeighter,
) -> None:
    """
    Make a plot of delta M before and after the correction

    """
    print("finding bins")
    bins = _bins()

    # Find ratio
    print("ratio before")
    dcs_count, dcs_err, cf_count, cf_err = _ratio_err(bins, year, magnetisation)
    ratio_before, err_before = ratio_err(dcs_count, cf_count, dcs_err, cf_err)

    print("ratio after")
    dcs_count, dcs_err, cf_count, cf_err = _ratio_err(
        bins,
        year,
        magnetisation,
        dcs_reweighter=dcs_reweighter,
        cf_reweighter=cf_reweighter,
        dcs_sum_before=np.sum(dcs_count),
        cf_sum_before=np.sum(cf_err),
    )
    ratio_after, err_after = ratio_err(dcs_count, cf_count, dcs_err, cf_err)

    # under/overflow bins
    bins = bins[1:-1]

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
        y=ratio_before,
        yerr=err_before,
        fmt="k.",
        label="Raw",
        **errorbar_kw,
    )
    axis["A"].errorbar(
        y=ratio_after, yerr=err_after, fmt="b.", label="Corrected", **errorbar_kw
    )

    # Plot the difference
    diff = ratio_after - ratio_before
    diff_err = np.sqrt(err_after**2 + err_before**2)

    axis["B"].axhline(0.0, color="k")
    axis["B"].errorbar(y=diff / diff_err, yerr=1.0, fmt="k.", **errorbar_kw)

    # Legends, labels, etc
    axis["A"].legend()

    axis["A"].set_ylabel(r"$\frac{WS}{RS}$")

    axis["B"].set_xlabel(r"$\Delta M$")
    axis["B"].set_ylabel(r"$\Delta$ count (normalised)")

    fig.suptitle(
        f"{year} {magnetisation} Efficiency correction\n Time ratio (no mass fit)"
    )

    fig.tight_layout()

    path = f"eff_times_{year}_{magnetisation}.png"
    print(f"saving {path}")
    fig.savefig(path)


def main(
    *,
    year: str,
    magnetisation: str,
    data_k_charge: str,
    weighter_k_charge: str,
    fit: bool,
    cut: bool,
):
    """
    Make plots of delta M and the phase space

    """
    # Generator of real data dataframes
    if cut:
        raise NotImplementedError("bdt cut the dfs")

    if data_k_charge != "both":
        raise NotImplementedError("haven't done this yet")

    # Open reweighters
    cf_reweighter = get_reweighter(
        year,
        "cf",
        magnetisation,
        weighter_k_charge,
        fit=fit,
        cut=cut,
        verbose=True,
    )
    dcs_reweighter = get_reweighter(
        year,
        "dcs",
        magnetisation,
        weighter_k_charge,
        fit=fit,
        cut=cut,
        verbose=True,
    )

    _plot_time_ratio(year, magnetisation, cf_reweighter, dcs_reweighter)


if __name__ == "__main__":
    parser = common.parser("Plot projections of phase space variables for real data")

    # Remove CF/DCS arguments, since we run over both DCS/CF
    # data and use both reweighters in this script
    common.remove_arg(parser, "decay_type")
    common.remove_arg(parser, "weighter_type")

    main(**vars(parser.parse_args()))
