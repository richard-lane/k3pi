"""
Plot a histogram of D0 IPCHI2

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
from lib_data import get, stats


def main():
    """
    Iterate over dataframes, filling a histogram of D0 IPCHI2

    """
    bins = np.concatenate(([-20], np.linspace(-10, 10, 100), [20]))
    counts, errors = stats.counts_generator(
        (
            np.log(dataframe["D0 ipchi2"])
            for dataframe in get.data("2018", "dcs", "magdown")
        ),
        bins,
    )

    plt.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        counts,
        xerr=(bins[1:] - bins[:-1]) / 2,
        yerr=errors,
        fmt="k.",
    )
    plt.show()


if __name__ == "__main__":
    main()
