"""
Plot histograms of RS and WS D0 IPCHI2, to look for secondaries

Plot them also at various values of the BDT cut to see how this affects them

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
from lib_data import get, stats
from lib_cuts.get import classifier as get_clf, cut_dfs
from lib_cuts.definitions import Classifier


def _time_indices(sign: str, time_bins: np.ndarray) -> np.ndarray:
    """
    Get time bin indices for 2018 MagDown

    """
    return stats.bin_indices(
        (dataframe["time"] for dataframe in get.data("2018", sign, "magdown")),
        time_bins,
    )


def _counts_no_cut(
    year: str,
    sign: str,
    magnetisation: str,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get counts in bins of D0 ipchi2

    """
    counts, errors = stats.counts_generator(
        (
            np.log(dataframe["D0 ipchi2"])
            for dataframe in get.data(year, sign, magnetisation)
        ),
        bins,
    )

    return counts, errors


def _counts(
    year: str,
    sign: str,
    magnetisation: str,
    bins: np.ndarray,
    clf: Classifier,
    threshhold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get counts in bins of D0 ipchi2

    """
    # Get a generator of dataframes after the BDT cut
    dataframes = cut_dfs(get.data(year, sign, magnetisation), clf, threshhold)

    counts, errors = stats.counts_generator(
        (np.log(dataframe["D0 ipchi2"]) for dataframe in dataframes),
        bins,
    )

    return counts, errors


def _plot_nocut_hists(
    axes: Tuple[plt.Axes, plt.Axes], year: str, magnetisation: str, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot histograms of ipchi2 with no cuts

    """
    centres = (bins[1:] + bins[:-1]) / 2

    dcs_counts_nocut, dcs_err_nocut = _counts_no_cut(year, "dcs", magnetisation, bins)
    cf_counts_nocut, cf_err_nocut = _counts_no_cut(year, "cf", magnetisation, bins)

    axes[0].errorbar(
        centres,
        dcs_counts_nocut,
        yerr=dcs_err_nocut,
        fmt="-",
        label="no cut",
        linewidth=0.75,
    )
    axes[1].errorbar(
        centres,
        cf_counts_nocut,
        yerr=cf_err_nocut,
        fmt="-",
        linewidth=0.75,
    )

    return dcs_counts_nocut, cf_counts_nocut


def _plot_cut_hists(
    axes: Tuple[plt.Axes, plt.Axes],
    year: str,
    magnetisation: str,
    bins: np.ndarray,
    clf: Classifier,
    threshhold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot hists at this BDT cut threshhold

    """
    centres = (bins[1:] + bins[:-1]) / 2

    dcs_counts, dcs_errors = _counts(
        year,
        "dcs",
        magnetisation,
        bins,
        clf,
        threshhold,
    )
    cf_counts, cf_errors = _counts(
        year,
        "cf",
        magnetisation,
        bins,
        clf,
        threshhold,
    )

    axes[0].errorbar(
        centres,
        dcs_counts,
        yerr=dcs_errors,
        fmt="-",
        label=f"{threshhold=}",
        linewidth=0.75,
    )
    axes[1].errorbar(
        centres,
        cf_counts,
        yerr=cf_errors,
        fmt="-",
        linewidth=0.75,
    )

    return dcs_counts, cf_counts


def main():
    """
    Iterate over dataframes, filling a histogram of D0 IPCHI2

    """
    # Get the BDT
    year, magnetisation = "2018", "magdown"
    clf = get_clf(year, "dcs", magnetisation)

    bins = np.concatenate(([-20], np.linspace(-10, 10, 500), [20]))
    centres = (bins[1:] + bins[:-1]) / 2

    # For the histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    dcs_count_nocut, cf_count_nocut = _plot_nocut_hists(axes, year, magnetisation, bins)

    # For the efficiencies
    eff_fig, eff_axes = plt.subplots(1, 2, figsize=(10, 5))

    for threshhold, colour in tqdm(
        zip(
            [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.20],
            plt.rcParams["axes.prop_cycle"].by_key()["color"][1:],
        )
    ):
        dcs_count, cf_count = _plot_cut_hists(
            axes, year, magnetisation, bins, clf, threshhold
        )

        eff_axes[0].plot(
            centres, dcs_count / dcs_count_nocut, label=f"{threshhold=}", color=colour
        )
        eff_axes[1].plot(centres, cf_count / cf_count_nocut, color=colour)

    for axis in axes:
        axis.set_xlim(-10, 10)
        axis.set_xlabel(r"log$_\mathrm{e}(\mathrm{D}^0\ \mathrm{IP}\chi^2)$")

    for axs in axes, eff_axes:
        axs[0].legend()
        axs[0].set_title("DCS")
        axs[1].set_title("CF")

    for axis in eff_axes:
        axis.set_xlim(-5, None)

    for figure in fig, eff_fig:
        figure.tight_layout()
    fig.suptitle(r"D0 IP$\chi^2$")
    eff_fig.suptitle(r"D0 IP$\chi^2$ BDT cut efficiency")

    fig.savefig("ipchi2_bdt_cuts.png")
    eff_fig.savefig("ipchi2_bdt_cut_efficiency.png")

    plt.show()


if __name__ == "__main__":
    main()
