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
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))

from libFit import pdfs, fit, util as mass_util
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS
from lib_cuts.get import cut_dfs
from lib_cuts.get import classifier as get_clf


def _plot(
    axes: Tuple[plt.Axes, plt.Axes],
    counts: np.ndarray,
    bins: np.ndarray,
    fit_params: Tuple,
) -> None:
    """
    Plot fit and pull on two axes

    """
    centres = (bins[1:] + bins[:-1]) / 2
    scale = np.sum(counts) * (bins[1] - bins[0])

    err_kw = {"fmt": "k.", "elinewidth": 0.5, "markersize": 1.0}
    axes[0].errorbar(
        centres,
        counts,
        yerr=np.sqrt(counts),
        **err_kw,
    )

    predicted = scale * pdfs.fractional_pdf(centres, *fit_params)
    axes[0].plot(centres, predicted)
    axes[0].plot(
        centres,
        scale * fit_params[0] * pdfs.normalised_signal(centres, *fit_params[1:-4]),
        label="signal",
    )
    axes[0].plot(
        centres,
        scale * (1 - fit_params[0]) * pdfs.normalised_bkg(centres, *fit_params[-4:]),
        label="bkg",
    )

    # Plot pull
    diff = counts - predicted
    axes[1].plot(pdfs.domain(), [1, 1], "r-")
    axes[1].errorbar(
        centres,
        diff,
        yerr=np.sqrt(counts),
        **err_kw,
    )


def _fit(
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
    _plot((axes["A"], axes["C"]), rs_count, bins, rs_params)
    _plot((axes["B"], axes["D"]), ws_count, bins, ws_params)

    axes["A"].legend()

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].plot(pdfs.domain(), [1, 1], "r-")

    fig.suptitle(f"{fitter.valid=}")

    fig.tight_layout()
    fig.savefig(f"{fit_dir}fit_{bin_number}.png")
    plt.close(fig)


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _generators(
    year: str, magnetisation: str, *, bdt_cut: bool
) -> Tuple[Iterable[pd.DataFrame], Iterable[pd.DataFrame]]:
    """
    Generator of rs/ws dataframes, with/without BDT cut

    """
    generators = get.data(year, "cf", magnetisation), get.data(
        year, "dcs", magnetisation
    )
    if bdt_cut:
        # Get the classifier
        clf = get_clf(year, "dcs", magnetisation)
        return [cut_dfs(gen, clf) for gen in generators]

    return generators


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
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=bdt_cut)

    dcs_indices = _time_indices(dcs_gen, time_bins)
    cf_indices = _time_indices(cf_gen, time_bins)

    # Need to get new generators now that we've used them up
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=bdt_cut)
    n_time_bins = len(time_bins) - 1
    dcs_counts, _ = mass_util.binned_delta_m_counts(
        dcs_gen, bins, n_time_bins, dcs_indices
    )
    cf_counts, _ = mass_util.binned_delta_m_counts(
        cf_gen, bins, n_time_bins, cf_indices
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
    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=True)

    fit_dir = "bdt_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    # Plot the fit in each bin
    for i, (dcs_count, cf_count) in enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1])):
        _fit(cf_count, dcs_count, i, bins, fit_dir)

    # Do the same without BDT cut
    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=False)
    fit_dir = "raw_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)
    for i, (dcs_count, cf_count) in enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1])):
        _fit(cf_count, dcs_count, i, bins, fit_dir)


if __name__ == "__main__":
    main()
