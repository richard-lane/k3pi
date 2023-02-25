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

from libFit import pdfs, fit, toy_utils, plotting, definitions, util
from lib_data import stats


def _gen():
    """
    Generate points
    """
    sign, time_bin = "cf", 5

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 2_000_000, 5_000_000

    rng = np.random.default_rng()
    sig = toy_utils.gen_sig(rng, n_sig, util.signal_param_guess(time_bin), verbose=True)
    bkg = toy_utils.gen_bkg_sqrt(
        rng, n_bkg, util.sqrt_bkg_param_guess(sign), verbose=True
    )

    return np.concatenate((sig, bkg))


def main():
    """
    just do 1 fit

    """
    combined = _gen()

    # Perform fits to the restricted range
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    counts, errs = stats.counts(combined, bins)

    # We don't want to fit to the stuff in the underflow bins
    binned_fitter = fit.binned_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        "cf",
        5,
        0.9,  # Signal frac guess
        (fit_low, gen_high),  # Fit to reduced region
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

    axes["B"].set_title("Alt bkg fit removed")

    axes["A"].set_title("Sqrt bkg fit")

    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig("toy_mass_fit.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy fit; sqrt model only for now")

    main(**vars(parser.parse_args()))
