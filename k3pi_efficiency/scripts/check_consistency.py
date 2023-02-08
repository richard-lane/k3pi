"""
Check consistency of the reweighter when trained on different training samples

Split the particle gun and MC into N+1 samples (one for testing, the rest for training)

Plots the derived time ratio after the reweighting for each of these reweighters on
the respective training sample and the testing sample

"""
import sys
import pathlib
import argparse
from typing import Tuple
from multiprocessing import Manager, Process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from lib_efficiency import efficiency_util
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_data import get, util, stats
from lib_time_fit.util import ratio_err


def _points(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Phase space, time points

    """
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)

    # Momentum order
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    return np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))


def _train_reweighter(
    ampgen_df: pd.DataFrame, pgun_df: pd.DataFrame, out_dict: dict, key: str
) -> None:
    """
    Train a reweighter, add it to the dictionary with the provided key

    """
    ampgen = _points(ampgen_df)
    pgun = _points(pgun_df)

    train_kwargs = {
        "n_estimators": 200,
        "max_depth": 3,
        # "learning_rate": 0.7,
        # "min_samples_leaf": 1800,
    }
    reweighter = EfficiencyWeighter(
        ampgen, pgun, fit=False, min_t=MIN_TIME, **train_kwargs
    )

    out_dict[key] = reweighter


def _add_train_indices(
    rng: np.random.Generator, dataframe: pd.DataFrame, num_trained: int
) -> None:
    """
    Overwrite or add a "train" column to the dataframe

    """
    dataframe["train"] = rng.integers(0, num_trained + 1, len(dataframe))


def _bins(max_time: float) -> np.ndarray:
    """Time bins"""
    return np.linspace(MIN_TIME, max_time, 15)


def _ratio_err(
    max_time: float,
    rs_df: pd.DataFrame,
    ws_df: pd.DataFrame,
    rs_wts: np.ndarray,
    ws_wts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error for plotting

    """
    bins = _bins(max_time)
    rs_counts, rs_err = stats.counts(rs_df["time"], bins, weights=rs_wts)
    ws_counts, ws_err = stats.counts(ws_df["time"], bins, weights=ws_wts)

    return ratio_err(ws_counts, rs_counts, ws_err, rs_err)


def _plot_time_ratio(
    axis: plt.Axes,
    max_time: float,
    rs_df: pd.DataFrame,
    ws_df: pd.DataFrame,
    rs_weighter: EfficiencyWeighter,
    ws_weighter: EfficiencyWeighter,
    label: str = None,
) -> None:
    """
    Plot the weighted ratio of WS to RS times

    """
    rs_wts = rs_weighter.weights(_points(rs_df)) if rs_weighter is not None else None
    ws_wts = ws_weighter.weights(_points(ws_df)) if ws_weighter is not None else None

    ratio, err = _ratio_err(max_time, rs_df, ws_df, rs_wts, ws_wts)

    time_bins = _bins(max_time)
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    widths = (time_bins[1:] - time_bins[:-1]) / 2

    # Plot both line and points
    line, _, _ = axis.errorbar(
        centres, ratio, xerr=widths, yerr=err, fmt="--", alpha=0.3
    )
    axis.plot(centres, ratio, "+", color=line.get_color(), label=label)


def _time_cut(dataframe: pd.DataFrame, max_time: float) -> pd.DataFrame:
    """
    Cut out times below MIN_TIME and above max_time

    """
    keep = (MIN_TIME < dataframe["time"]) & (dataframe["time"] < max_time)

    return dataframe[keep]


def main(num_trained: int):
    """
    Get the pgun and ampgen data, split it into the right number of chunks

    Train the reweighters (in parallel) on the chunks; bring them back together and plot the ratio

    """
    # Get the pgun and ampgen dataframes
    rs_pgun_df = get.particle_gun("cf")
    ws_pgun_df = get.particle_gun("dcs")

    rs_ampgen_df = get.ampgen("cf")
    ws_ampgen_df = get.ampgen("dcs")

    # Time cuts
    max_time = 10.0
    rs_pgun_df = _time_cut(rs_pgun_df, max_time)
    ws_pgun_df = _time_cut(ws_pgun_df, max_time)
    rs_ampgen_df = _time_cut(rs_ampgen_df, max_time)
    ws_ampgen_df = _time_cut(ws_ampgen_df, max_time)

    # Assign random numbers to the dataframes
    # 0 is for testing; training split otherwise
    rng = np.random.default_rng()

    _add_train_indices(rng, rs_pgun_df, num_trained)
    _add_train_indices(rng, ws_pgun_df, num_trained)

    _add_train_indices(rng, rs_ampgen_df, num_trained)
    _add_train_indices(rng, ws_ampgen_df, num_trained)

    # Train reweighters on each of the splits
    manager = Manager()
    ws_reweighters = manager.dict()
    rs_reweighters = manager.dict()

    ws_procs = [
        Process(
            target=_train_reweighter,
            args=(
                ws_ampgen_df[ws_ampgen_df["train"] == i],
                ws_pgun_df[ws_pgun_df["train"] == i],
                ws_reweighters,
                i,
            ),
        )
        for i in range(1, num_trained + 1)
    ]
    rs_procs = [
        Process(
            target=_train_reweighter,
            args=(
                rs_ampgen_df[rs_ampgen_df["train"] == i],
                rs_pgun_df[rs_pgun_df["train"] == i],
                rs_reweighters,
                i,
            ),
        )
        for i in range(1, num_trained + 1)
    ]

    for proc in ws_procs:
        proc.start()
    for proc in rs_procs:
        proc.start()

    for proc in ws_procs:
        proc.join()
    for proc in rs_procs:
        proc.join()

    # Plot unweighted ratio
    fig, axis = plt.subplots()
    _plot_time_ratio(
        axis,
        max_time,
        rs_pgun_df,
        ws_pgun_df,
        None,
        None,
        "Unweighted (all)",
    )

    # Plot time ratios for training samples
    for i in range(1, num_trained + 1):
        _plot_time_ratio(
            axis,
            max_time,
            rs_pgun_df[rs_pgun_df["train"] == i],
            ws_pgun_df[ws_pgun_df["train"] == i],
            rs_reweighters[i],
            ws_reweighters[i],
            i,
        )

    axis.legend()
    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"$\frac{\mathrm{WS}}{\mathrm{RS}}$")
    fig.suptitle("training")
    fig.tight_layout()
    fig.savefig("efficiency_consistency_train.png")
    plt.close(fig)

    # Plot time ratios for the testing samples
    fig, axis = plt.subplots()
    _plot_time_ratio(
        axis,
        max_time,
        rs_pgun_df[rs_pgun_df["train"] == 0],
        ws_pgun_df[ws_pgun_df["train"] == 0],
        None,
        None,
        "Unweighted (test)",
    )

    # Plot time ratios for testing sample
    for i in range(1, num_trained + 1):
        _plot_time_ratio(
            axis,
            max_time,
            rs_pgun_df[rs_pgun_df["train"] == 0],
            ws_pgun_df[ws_pgun_df["train"] == 0],
            rs_reweighters[i],
            ws_reweighters[i],
            i,
        )

    axis.legend()
    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"$\frac{\mathrm{WS}}{\mathrm{RS}}$")
    fig.suptitle("testing")
    fig.tight_layout()
    fig.savefig("efficiency_consistency_test.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train several reweighters; check consistency between them"
    )
    parser.add_argument(
        "--num_trained",
        "-n",
        type=int,
        help="number of training splits to make",
        required=True,
    )

    main(**vars(parser.parse_args()))
