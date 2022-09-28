"""
Simultaneous fit to RS and WS without cuts

"""
import sys
import glob
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process

import tqdm
import uproot

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from libFit import pdfs
from libFit import fit
from lib_data import get


def _delta_m(year: str, sign: str, magnetisation: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delta M arrays and times

    """
    df = pd.concat(get.data(year, sign, magnetisation))

    return df["D* mass"] - df["D0 mass"], df["time"]


def _plot(rs: np.ndarray, ws: np.ndarray, params: tuple, bin_number: int) -> None:
    """
    Plot the fit

    """
    rs_params = params[:-1]
    ws_params = (params[-1], *params[1:-1])

    fig, ax = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(12, 8)
    )

    bins = np.linspace(*pdfs.domain(), 250)
    centres = (bins[1:] + bins[:-1]) / 2

    rs_counts, _ = np.histogram(rs, bins)
    ws_counts, _ = np.histogram(ws, bins)

    rs_err = np.sqrt(rs_counts)
    ws_err = np.sqrt(ws_counts)

    rs_scale = len(rs) * (bins[1] - bins[0])
    ws_scale = len(ws) * (bins[1] - bins[0])

    ax["A"].errorbar(centres, rs_counts, yerr=rs_err, fmt="k.")
    ax["B"].errorbar(centres, ws_counts, yerr=ws_err, fmt="k.")

    rs_predicted = rs_scale * pdfs.fractional_pdf(centres, *rs_params)
    ws_predicted = ws_scale * pdfs.fractional_pdf(centres, *ws_params)

    ax["A"].plot(centres, rs_predicted)
    ax["B"].plot(centres, ws_predicted)

    ax["A"].plot(
        centres,
        rs_scale * rs_params[0] * pdfs.normalised_signal(centres, *rs_params[1:-2]),
        label="signal",
    )
    ax["B"].plot(
        centres,
        ws_scale * ws_params[0] * pdfs.normalised_signal(centres, *ws_params[1:-2]),
        label="signal",
    )

    ax["A"].plot(
        centres,
        rs_scale * (1 - rs_params[0]) * pdfs.normalised_bkg(centres, *rs_params[-2:]),
        label="bkg",
    )
    ax["B"].plot(
        centres,
        ws_scale * (1 - ws_params[0]) * pdfs.normalised_bkg(centres, *ws_params[-2:]),
        label="bkg",
    )

    ax["A"].legend()

    rs_diff = rs_counts - rs_predicted
    ws_diff = ws_counts - ws_predicted

    ax["C"].plot(pdfs.domain(), [1, 1], "r-")
    ax["D"].plot(pdfs.domain(), [1, 1], "r-")

    ax["C"].errorbar(centres, rs_diff, yerr=rs_err, fmt="k.")
    ax["D"].errorbar(centres, ws_diff, yerr=ws_err, fmt="k.")

    fig.tight_layout()
    fig.savefig(f"simultaneous_{bin_number}.png")

    plt.show()


def _prune(delta_m: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove masses outside the fit range and negative/NaN times

    """
    low, high = pdfs.domain()

    keep = delta_m < high
    print(f"{np.sum(keep)} : {len(keep)}")

    keep &= delta_m > low
    print(f"{np.sum(keep)} : {len(keep)}")

    keep &= times > 0
    print(f"{np.sum(keep)} : {len(keep)}")

    keep &= np.isfinite(times)
    print(f"{np.sum(keep)} : {len(keep)}")

    return delta_m[keep], times[keep]


def _plot_hists(
    ax: np.ndarray, delta_m: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot but don't show or save histograms of mass difference + times

    """
    ax[0].hist(
        delta_m, bins=np.linspace(*pdfs.domain(), 100), density=True, histtype="step"
    )

    ax[1].hist(times, bins=np.linspace(0, 8.0, 100), density=True, histtype="step")


def _fit_and_plot(rs, ws, bin_number):
    """
    Fit and plot delta m dist in a time bin

    """
    fitter = fit.simultaneous_fit(rs, ws, bin_number)  # time bin 5 for now

    _plot(rs, ws, fitter.values, bin_number)


def main():
    rs, rs_t = _prune(*_delta_m("2018", "cf", "magdown"))
    ws, ws_t = _prune(*_delta_m("2018", "dcs", "magdown"))

    d_lifetime = 0.41  # picoseconds
    time_bins = tuple(
        x * d_lifetime
        for x in (0.0, 0.94, 1.185, 1.40, 1.62, 1.85, 2.13, 2.45, 2.87, 3.5, 8.0, 19.0)
    )
    rs_t_indices = np.digitize(rs_t, time_bins)
    ws_t_indices = np.digitize(ws_t, time_bins)

    _, ax = plt.subplots(1, 2)
    for i in np.unique(rs_t_indices):
        #     # Plot masses and times
        _plot_hists(ax, rs[rs_t_indices == i], rs_t[rs_t_indices == i])
    #     _plot_hists(ax, ws[ws_t_indices == i], ws_t[ws_t_indices == i])
    plt.show()

    # procs = [
    #     Process(
    #         target=_fit_and_plot, args=(rs[rs_t_indices == i], ws[ws_t_indices == i], i)
    #     )
    #     for i in np.unique(rs_t_indices)
    # ]

    # for p in procs:
    #     p.start()
    # for p in procs:
    #     p.join()
    _fit_and_plot(rs, ws, 5)


if __name__ == "__main__":
    main()
