"""
Simultaneous fit to RS and WS without cuts

"""
import os
import sys
import pathlib
from typing import Tuple, Iterable, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))

from libFit import pdfs, fit, util as mass_util
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS


def _plot(
    rs_count: np.ndarray,
    ws_count: np.ndarray,
    bin_number: int,
    bins: np.ndarray,
    fit_dir: str,
) -> None:
    """
    Plot the fit

    """
    fitter = fit.binned_simultaneous_fit(rs_count, ws_count, bins, bin_number)
    params = fitter.values
    print(f"{fitter.valid=}", end="\t")
    print(f"{params[-2]} +- {fitter.errors[-2]}", end="\t")
    print(f"{params[-1]} +- {fitter.errors[-1]}")

    rs_params = (params[0], *params[2:])
    ws_params = tuple(params[1:])

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(12, 8)
    )

    centres = (bins[1:] + bins[:-1]) / 2
    rs_scale = np.sum(rs_count) * (bins[1] - bins[0])
    ws_scale = np.sum(ws_count) * (bins[1] - bins[0])

    axes["A"].errorbar(
        centres,
        rs_count,
        yerr=np.sqrt(rs_count),
        fmt="k.",
        elinewidth=0.5,
        markersize=1.0,
    )
    axes["B"].errorbar(
        centres,
        ws_count,
        yerr=np.sqrt(ws_count),
        fmt="k.",
        elinewidth=0.5,
        markersize=1.0,
    )

    rs_predicted = rs_scale * pdfs.fractional_pdf(centres, *rs_params)
    ws_predicted = ws_scale * pdfs.fractional_pdf(centres, *ws_params)

    axes["A"].plot(centres, rs_predicted)
    axes["B"].plot(centres, ws_predicted)

    axes["A"].plot(
        centres,
        rs_scale * rs_params[0] * pdfs.normalised_signal(centres, *rs_params[1:-4]),
        label="signal",
    )
    axes["B"].plot(
        centres,
        ws_scale * ws_params[0] * pdfs.normalised_signal(centres, *ws_params[1:-4]),
        label="signal",
    )

    axes["A"].plot(
        centres,
        rs_scale * (1 - rs_params[0]) * pdfs.normalised_bkg(centres, *rs_params[-4:]),
        label="bkg",
    )
    axes["B"].plot(
        centres,
        ws_scale * (1 - ws_params[0]) * pdfs.normalised_bkg(centres, *ws_params[-4:]),
        label="bkg",
    )

    axes["A"].legend()

    rs_diff = rs_count - rs_predicted
    ws_diff = ws_count - ws_predicted

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].plot(pdfs.domain(), [1, 1], "r-")

    axes["C"].errorbar(
        centres,
        rs_diff,
        yerr=np.sqrt(rs_count),
        fmt="k.",
        elinewidth=0.5,
        markersize=1.0,
    )
    axes["D"].errorbar(
        centres,
        ws_diff,
        yerr=np.sqrt(ws_count),
        fmt="k.",
        elinewidth=0.5,
        markersize=1.0,
    )

    fig.suptitle(f"{fitter.valid=}")

    fig.tight_layout()
    fig.savefig(f"{fit_dir}fit_{bin_number}.png")


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _counts(
    year: str,
    magnetisation: str,
    bins: np.ndarray,
    time_bins: np.ndarray,
    *,
    bdt_cut: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of counts/errors in each time bin

    """
    # Get generators of time indices
    dcs_indices = _time_indices(get.data(year, "dcs", magnetisation), time_bins)
    cf_indices = _time_indices(get.data(year, "cf", magnetisation), time_bins)

    n_time_bins = len(time_bins) - 1
    dcs_counts, _ = mass_util.binned_delta_m_counts(
        get.data(year, "dcs", magnetisation), bins, n_time_bins, dcs_indices
    )
    cf_counts, _ = mass_util.binned_delta_m_counts(
        get.data(year, "cf", magnetisation), bins, n_time_bins, cf_indices
    )

    return dcs_counts, cf_counts


def main():
    """
    Do mass fits in each time bin without BDT cuts

    """
    bins = np.linspace(*pdfs.domain(), 400)
    time_bins = np.array((-100, *TIME_BINS[1:], 100))

    # Get delta M values from a generator of dataframes
    year, magnetisation = "2018", "magdown"
    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=False)

    fit_dir = "raw_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    # Plot the fit in each bin
    for i, (dcs_count, cf_count) in enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1])):
        _plot(cf_count, dcs_count, i, bins, fit_dir)


if __name__ == "__main__":
    main()
