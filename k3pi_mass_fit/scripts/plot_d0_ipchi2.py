"""
Plot a histogram of D0 IPCHI2

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS


def _time_indices(sign: str, time_bins: np.ndarray) -> np.ndarray:
    """
    Get time bin indices for 2018 MagDown

    """
    return stats.bin_indices(
        (dataframe["time"] for dataframe in get.data("2018", sign, "magdown")),
        time_bins,
    )


def _counts(
    sign: str,
    time_bins: np.ndarray,
    bins: np.ndarray,
    time_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get counts in bins of D0 ipchi2 for 2018 MagDown

    """
    counts, errors = stats.time_binned_counts(
        (
            np.log(dataframe["D0 ipchi2"])
            for dataframe in get.data("2018", sign, "magdown")
        ),
        bins,
        len(time_bins) - 1,
        time_indices,
    )

    return counts, errors


def main():
    """
    Iterate over dataframes, filling a histogram of D0 IPCHI2

    """
    # Get time indices
    time_bins = np.array((-100, *TIME_BINS[1:], np.inf))

    # Get counts
    bins = np.concatenate(([-15], np.linspace(-10, 10, 500), [15]))
    dcs_counts, dcs_errors = _counts(
        "dcs", time_bins, bins, _time_indices("dcs", time_bins)
    )
    cf_counts, cf_errors = _counts(
        "cf", time_bins, bins, _time_indices("cf", time_bins)
    )

    # Change colour cycle to have enough colours
    fig, axes = plt.subplots(1, 2)
    colours = list(plt.cm.tab10(np.arange(10))) + ["plum", "crimson"]
    for axis in axes:
        axis.set_prop_cycle("color", colours)

    centres = (bins[1:] + bins[:-1]) / 2
    for i, (dcs_count, dcs_error, cf_count, cf_error) in enumerate(
        zip(dcs_counts[1:-1], dcs_errors[1:-1], cf_counts[1:-1], cf_errors[1:-1])
    ):
        axes[0].errorbar(
            centres,
            dcs_count,
            yerr=dcs_error,
            fmt="-",
            label=f"time bin {i}",
            linewidth=0.75,
        )
        axes[1].errorbar(
            centres,
            cf_count,
            yerr=cf_error,
            fmt="-",
            label=f"time bin {i}",
            linewidth=0.75,
        )

    for axis in axes:
        axis.set_xlim(-10, 10)
        axis.set_xlabel(r"log$_\mathrm{e}(\mathrm{D}^0\ \mathrm{IP}\chi^2)$")

    axes[0].legend()
    axes[0].set_title("DCS")
    axes[1].set_title("CF")

    plt.show()


if __name__ == "__main__":
    main()
