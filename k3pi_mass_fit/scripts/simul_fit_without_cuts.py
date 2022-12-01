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

from libFit import pdfs, fit, util as mass_util, plotting
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS
from lib_cuts.get import cut_dfs
from lib_cuts.get import classifier as get_clf


def _separate_fit(
    count: np.ndarray, bin_number: int, bins: np.ndarray, sign: str, plot_path: str
):
    """
    Plot separate fits

    """
    fitter = fit.binned_fit(
        count, bins, sign, bin_number, 0.9 if sign == "RS" else 0.05
    )

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(6, 8))
    plotting.mass_fit(
        (axes["A"], axes["B"]), count, np.sqrt(count), bins, fitter.values
    )

    axes["A"].legend()

    axes["B"].plot(pdfs.domain(), [1, 1], "r-")

    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()

    fig.tight_layout()
    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)


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

    fig, _ = plotting.simul_fits(
        rs_count, np.sqrt(rs_count), ws_count, np.sqrt(ws_count), bins, params
    )

    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()

    plot_path = f"{fit_dir}fit_{bin_number}.png"
    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
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


def _mkdirs(base: str) -> None:
    """
    Make plot dirs

    """
    rs_dir = os.path.join(base, "rs/")
    ws_dir = os.path.join(base, "ws/")
    for dir_ in base, rs_dir, ws_dir:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)


def main():
    """
    Do mass fits in each time bin without BDT cuts

    """
    bins = np.linspace(*pdfs.domain(), 400)
    time_bins = np.array((-100, *TIME_BINS[1:], 100))

    # Get delta M values from a generator of dataframes
    year, magnetisation = "2018", "magdown"
    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=True)

    bdt_dir = "bdt_fits/"
    _mkdirs(bdt_dir)

    # Plot the fit in each bin
    for i, (dcs_count, cf_count) in enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1])):
        _fit(cf_count, dcs_count, i, bins, bdt_dir)
        _separate_fit(cf_count, i, bins, "RS", f"{bdt_dir}rs/{i}.png")
        _separate_fit(dcs_count, i, bins, "WS", f"{bdt_dir}ws/{i}.png")

    # Do the same without BDT cut
    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=False)

    raw_dir = "raw_fits/"
    _mkdirs(raw_dir)

    for i, (dcs_count, cf_count) in enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1])):
        _fit(cf_count, dcs_count, i, bins, raw_dir)
        _separate_fit(cf_count, i, bins, "RS", f"{raw_dir}rs/{i}.png")
        _separate_fit(dcs_count, i, bins, "WS", f"{raw_dir}ws/{i}.png")


if __name__ == "__main__":
    main()
