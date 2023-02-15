"""
Fit to D0 IPCHI2 histogram

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
from lib_data import get, stats, ipchi2_fit


def main():
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    low, high = ipchi2_fit.domain()
    bins = np.concatenate(
        ([-np.inf], np.linspace(low, high * 0.9, 150), [high, np.inf])
    )

    # Get counts in the bins
    counts, errs = stats.counts_generator(
        (
            np.log(dataframe["D0 ipchi2"])
            for dataframe in get.data("2018", "cf", "magdown")
        ),
        bins,
    )
    # Remove the first and last counts and bins
    bins = bins[1:-1]
    counts = counts[1:-1]
    errs = errs[1:-1]

    # fit
    sig_defaults = {
        "centre_sig": 0.9,
        "width_l_sig": 1.6,
        "width_r_sig": 0.9,
        "alpha_l_sig": 0.0,
        "alpha_r_sig": 0.0,
        "beta_sig": 0.0,
    }
    bkg_defaults = {
        "centre_bkg": 4.5,
        "width_l_bkg": 1.8,
        "width_r_bkg": 1.8,
        "alpha_l_bkg": 0.0,
        "alpha_r_bkg": 0.0,
        "beta_bkg": 0.0,
    }
    sig_fraction = 0.85
    fitter = ipchi2_fit.fit(
        counts, bins, sig_fraction, sig_defaults, bkg_defaults, errors=errs
    )
    print(fitter)

    # plot
    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(8, 10))
    ipchi2_fit.plot(
        (axes["A"], axes["B"]), bins, counts, np.sqrt(counts), fitter.values
    )
    axes["A"].legend()

    fig.savefig("data_ipchi2_fit.png")


if __name__ == "__main__":
    main()
