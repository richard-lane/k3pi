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
    sqrt_fitter = fit.binned_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        initial_guess,
        (fit_low, gen_high),  # Fit to reduced region
    )
    alt_bkg_fitter = fit.alt_bkg_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        "2018",
        "magdown",
        "dcs",
        (*initial_guess[:8], 0, 0, 0),
        bdt_cut=False,
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
        sqrt_fitter.values,
    )
    plotting.alt_bkg_fit(
        (axes["B"], axes["D"]),
        counts,
        errs,
        bins,
        (fit_low, gen_high),
        alt_bkg_fitter.values,
        year="2018",
        magnetisation="magdown",
        sign="dcs",
        bdt_cut=False,
    )

    axes["B"].set_title("Alt bkg fit")

    axes["A"].set_title("Sqrt bkg fit")

    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig("toy_mass_fit.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Toy fits with the sqrt and alt bkg. Generates with the sqrt bkg"
    )

    main(**vars(parser.parse_args()))
