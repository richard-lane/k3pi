"""
Plot a histogram of D0 IPCHI2

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
from lib_data import get, stats
from lib_time_fit.definitions import TIME_BINS


def main():
    """
    Iterate over dataframes, filling a histogram of D0 IPCHI2

    """
    # Get time indices
    time_bins = np.array((-100, *TIME_BINS[1:], 100))
    time_indices = stats.bin_indices(
        (dataframe["time"] for dataframe in get.data("2018", "dcs", "magdown")),
        time_bins,
    )

    # Get counts
    bins = np.concatenate(([-15], np.linspace(-10, 10, 100)))
    counts, errors = stats.time_binned_counts(
        (
            np.log(dataframe["D0 ipchi2"])
            for dataframe in get.data("2018", "dcs", "magdown")
        ),
        bins,
        len(time_bins) - 1,
        time_indices,
    )

    # Change colour cycle to have enough colours
    fig, axis = plt.subplots()
    colours = list(plt.cm.tab10(np.arange(10))) + ["plum", "crimson"]
    axis.set_prop_cycle("color", colours)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    for i, (count, error) in enumerate(zip(counts[1:-1], errors[1:-1])):
        axis.errorbar(
            centres,
            count,
            xerr=widths,
            yerr=error,
            fmt="-",
            label=f"time bin {i}",
        )
    axis.set_xlim(-10, 10)
    axis.set_xlabel(r"log$_\mathrm{e}(\mathrm{D}^0\ \mathrm{IP}\chi^2)$")

    axis.legend()
    plt.show()


if __name__ == "__main__":
    main()
