"""
Do some mass fits after the BDT cut,
then use these to calculate sWeights

Use these sWeights to plot the distribution
of D0 IPchi2, and maybe also do a fit to them

"""
import sys
import pathlib
from typing import Tuple, Iterable, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))

from libFit import pdfs, fit, util as mass_util, sweighting, definitions
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS
from lib_cuts.get import cut_dfs
from lib_cuts.get import classifier as get_clf


def _generator(
    year: str,
    magnetisation: str,
    sign: str,
) -> Tuple[Iterable[pd.DataFrame], Iterable[pd.DataFrame]]:
    """
    Generator of rs/ws dataframes, with BDT cut

    """
    generator = get.data(year, sign, magnetisation)

    # Get the classifier
    # BDT cut is always with DCS
    clf = get_clf(year, "dcs", magnetisation)
    return cut_dfs(generator, clf)


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _mass_counts(
    year: str,
    magnetisation: str,
    sign: str,
    bins: np.ndarray,
    time_bins: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns a list of delta M counts/errors in each time bin

    """
    # Get generators of time indices
    gen = _generator(year, magnetisation, sign)

    indices = _time_indices(gen, time_bins)

    # Need to get new generators now that we've used them up
    gen = _generator(year, magnetisation, sign)
    n_time_bins = len(time_bins) - 1
    counts, _ = mass_util.binned_delta_m_counts(gen, bins, n_time_bins, indices)

    return counts


def _mass_fitters(
    year: str, magnetisation: str, mass_bins: np.ndarray, time_bins: np.ndarray
) -> List:
    """
    Do mass fit, find sWeighting fcns from PDFs, return the fitters

    """
    # Find counts
    dcs_counts = _mass_counts(year, magnetisation, "dcs", mass_bins, time_bins)
    cf_counts = _mass_counts(year, magnetisation, "cf", mass_bins, time_bins)

    fitters = []
    for time_bin, (dcs_count, cf_count) in tqdm(
        enumerate(zip(dcs_counts[1:-1], cf_counts[1:-1]))
    ):
        # Perform a mass fit
        fitter = fit.binned_simultaneous_fit(cf_count, dcs_count, mass_bins, time_bin)
        fitters.append(fitter)

    return fitters


def _ipchi2_counts(
    year: str,
    magnetisation: str,
    sign: str,
    ip_bins: np.ndarray,
    time_bins: np.ndarray,
):
    """
    unweighted ipchi2 counts in each time bin

    """
    # Get a generator of time bin indices
    time_indices = _time_indices(_generator(year, magnetisation, sign), time_bins)

    # Get a generator of arrays for log D0 ipchi2
    ip_gen = (
        np.log(dataframe["D0 ipchi2"])
        for dataframe in _generator(year, magnetisation, sign)
    )

    # Find the time-binned unweighted IPchi2 counts
    binned_counts = stats.time_binned_counts(
        ip_gen, ip_bins, len(time_bins) - 1, time_indices
    )

    return binned_counts


def _weighted_ipchi2_counts(
    year: str,
    magnetisation: str,
    sign: str,
    ip_bins: np.ndarray,
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
):
    """
    ipchi2 counts in each time bin with the sWeighting

    """
    # Get mass fitters
    fitters = _mass_fitters(year, magnetisation, mass_bins, time_bins)

    # Get params from these fitters
    params = (
        [tuple(fitter.values[2:]) for fitter in fitters]
        if sign == "dcs"
        else [(*fitter.values[:2], fitter.values[4:]) for fitter in fitters]
    )
    # Get sWeights from these params
    # For now just use the sWeights from the first time bin
    # TODO properly - sWeighting in each time bin
    sweights = sweighting.sweights(_generator(year, magnetisation, sign), params[0])

    # Get a generator of time bin indices
    time_indices = _time_indices(_generator(year, magnetisation, sign), time_bins)

    # Get a generator of arrays for log D0 ipchi2
    ip_gen = (
        np.log(dataframe["D0 ipchi2"])
        for dataframe in _generator(year, magnetisation, sign)
    )

    # Find the weighted counts in the bins
    return stats.time_binned_counts(
        ip_gen, ip_bins, len(time_bins) - 1, time_indices, sweights
    )


def main():
    """
    Do mass fits in each time bin without BDT cuts

    """
    mass_bins = definitions.mass_bins(200)
    time_bins = np.array((-np.inf, *TIME_BINS[1:], np.inf))
    ipchi2_bins = np.concatenate(([-20], np.linspace(-10, 10, 100), [20]))

    year, magnetisation = "2018", "magdown"

    # Plot DCS and CF ipchi2 with/without these sweighting fcns
    dcs_weighted_count, _ = _weighted_ipchi2_counts(
        year, magnetisation, "dcs", ipchi2_bins, time_bins, mass_bins
    )
    cf_weighted_count, _ = _weighted_ipchi2_counts(
        year, magnetisation, "cf", ipchi2_bins, time_bins, mass_bins
    )

    dcs_binned_count, _ = _ipchi2_counts(
        year, magnetisation, "dcs", ipchi2_bins, time_bins
    )
    cf_binned_count, _ = _ipchi2_counts(
        year,
        magnetisation,
        "cf",
        ipchi2_bins,
        time_bins,
    )

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    centres = (ipchi2_bins[1:] + ipchi2_bins[:-1]) / 2
    for i, (dcs, cf, dcs_sw, cf_sw) in enumerate(
        zip(
            dcs_binned_count[1:-1],
            cf_binned_count[1:-1],
            dcs_weighted_count[1:-1],
            cf_weighted_count[1:-1],
        )
    ):
        axes[0, 0].plot(centres, dcs, label=f"Time bin {i}")
        axes[0, 1].plot(centres, cf, label=f"Time bin {i}")

        axes[1, 0].plot(centres, dcs_sw)
        axes[1, 1].plot(centres, cf_sw)

    axes[0, 0].set_title("WS")
    axes[0, 1].set_title("RS")
    axes[0, 1].legend()
    for axis in axes[1]:
        axis.set_xlabel(r"log$_\mathrm{e}(\mathrm{D}^0\ \mathrm{IP}\chi^2)$")
    fig.savefig("ipchi2_with_bdtcut.png")
    plt.show()


if __name__ == "__main__":
    main()
