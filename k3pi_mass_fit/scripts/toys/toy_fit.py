"""
Generate some points with accept/reject, fit to them and show the fit

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


def _toy_fit():
    """
    Generate some stuff, do a fit to it

    """
    sign, time_bin = "cf", 5

    bkg_params = pdfs.background_defaults(sign)

    # Generate points along the whole range
    (gen_low, gen_high) = pdfs.domain()

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 2_000_000, 5_000_000
    combined, true_params = toy_utils.gen_points(
        np.random.default_rng(),
        n_sig,
        n_bkg,
        (gen_low, gen_high),
        bkg_params=bkg_params,
        verbose=True,
    )

    # Perform fits to the restricted range
    fit_low, _ = pdfs.reduced_domain()
    sig_frac = true_params[0] / len(combined)
    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    counts, errs = stats.counts(combined, bins)

    # We don't want to fit to the stuff in the underflow bins
    binned_fitter = fit.binned_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        sign,
        time_bin,
        sig_frac,
        (fit_low, gen_high),
    )

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", figsize=(18, 12), sharex=True
    )
    plotting.mass_fit(
        (axes["A"], axes["C"]),
        counts,
        errs,
        bins,
        (fit_low, gen_high),
        binned_fitter.values,
    )

    try:
        # Removed for now
        # alt_fitter = fit.alt_bkg_fit(
        #     counts, bins, sign, time_bin, sig_frac, bdt_cut=False, efficiency=False
        # )
        # plotting.alt_bkg_fit(
        #     (axes["B"], axes["D"]),
        #     counts,
        #     errs,
        #     bins,
        #     alt_fitter.values,
        #     sign=sign,
        #     bdt_cut=False,
        #     efficiency=False,
        # )
        # axes["B"].set_title("Alt bkg fit")
        axes["B"].set_title("Alt bkg fit (removed)")
    except FileNotFoundError:
        axes["B"].set_title("Alt bkg not possible; pickle dump not created")

    axes["A"].set_title("Sqrt bkg fit")

    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig("toy_mass_fit.png")


def main():
    """
    just do 1 fit

    """
    _toy_fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy fit; sqrt model only for now")

    main(**vars(parser.parse_args()))
