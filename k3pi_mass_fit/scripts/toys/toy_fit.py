"""
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, plotting, definitions, util
from lib_data import stats


def _gen(domain: Tuple[float, float]) -> Tuple[np.ndarray, Tuple]:
    """
    Generate points

    returns the points and a best guess at the params

    Generates points along all of pdfs.domain(),
    but returns the expected amount in the region given by
    the domain parameter

    """
    sign, time_bin = "cf", 5

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 2_000_000, 5_000_000

    sig_params = util.signal_param_guess(time_bin)
    bkg_params = util.sqrt_bkg_param_guess(sign)

    rng = np.random.default_rng()
    sig = toy_utils.gen_sig(rng, n_sig, sig_params, verbose=True)
    bkg = toy_utils.gen_bkg_sqrt(rng, n_bkg, bkg_params, verbose=True)

    # Number we expect to generate
    n_sig = toy_utils.n_expected_sig(n_sig, domain, sig_params)
    n_bkg = toy_utils.n_expected_bkg(n_bkg, domain, bkg_params)

    return np.concatenate((sig, bkg)), (n_sig, n_bkg, *sig_params, *bkg_params)


def main():
    """
    just do 1 fit

    """
    # Perform fits to the restricted range
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    combined, initial_guess = _gen((fit_low, gen_high))

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    counts, errs = stats.counts(combined, bins)

    # We don't want to fit to the stuff in the underflow bins
    binned_fitter = fit.binned_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        initial_guess,
        (fit_low, gen_high),  # Fit to reduced region
    )

    print("initial guess:", initial_guess)
    print("fit vals:", binned_fitter.values)

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
