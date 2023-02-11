"""
Generate some points with accept/reject, fit to them and show the fit
with the reduced domain

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, plotting, definitions
from lib_data import stats


def main():
    """
    Generate some stuff, do a fit to it

    """
    sign, time_bin = "cf", 5

    bkg_params = pdfs.background_defaults(sign)

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 2_000_000, 5_000_000
    combined, true_params = toy_utils.gen_points_reduced(
        np.random.default_rng(),
        n_sig,
        n_bkg,
        sign,
        time_bin,
        bkg_params,
    )

    # Perform fits
    sig_frac = true_params[0]
    bins = definitions.reduced_mass_bins(500)
    counts, errs = stats.counts(combined, bins)
    binned_fitter = fit.binned_fit_reduced(counts, bins, sign, time_bin, sig_frac)

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(12, 8))
    plotting.mass_fit_reduced(
        (axes["A"], axes["B"]), counts, errs, bins, binned_fitter.values
    )

    axes["A"].set_title("Sqrt bkg fit")

    fig.suptitle(f"toy data")
    fig.tight_layout()

    plt.savefig("toy_mass_fit_reduced.png")


if __name__ == "__main__":
    main()
