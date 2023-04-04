"""
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import argparse
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, plotting, definitions, util, bkg as lib_bkg
from lib_data import stats


def _gen(
    bins: np.ndarray, alt_bkg: bool, domain: Tuple[float, float], bkg_pdf: Callable
) -> Tuple[np.ndarray, Tuple]:
    """
    Generate points

    returns the points and a best guess at the params
    for both the sqrt and alt bkg fitters

    Generates points along all of pdfs.domain(),
    but returns the expected amount in the region given by
    the domain parameter

    """
    sign, time_bin = "cf", 5

    n_sig, n_bkg = 3_000, 25_000

    sig_params = util.signal_param_guess(time_bin)
    sqrt_bkg_params = util.sqrt_bkg_param_guess(sign)
    alt_bkg_params = (0, 0, 0)

    rng = np.random.default_rng()
    sig = toy_utils.gen_sig(rng, n_sig, sig_params, domain, verbose=True)
    if not alt_bkg:
        bkg = toy_utils.gen_bkg_sqrt(rng, n_bkg, sqrt_bkg_params, domain, verbose=True)
    else:
        bkg = toy_utils.gen_alt_bkg(rng, n_bkg, bkg_pdf, alt_bkg_params, domain)

    return (
        np.concatenate((sig, bkg)),
        (n_sig, n_bkg, *sig_params, *sqrt_bkg_params),
        (n_sig, n_bkg, *sig_params, *alt_bkg_params),
    )


def main(*, alt_bkg: bool):
    """
    just do 1 fit

    """
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    # Define bins
    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, gen_high), (n_underflow, 150)
    )

    # Define the alt bkg pdf in the fit region
    # Hard coded options for now
    bkg_pdf = lib_bkg.pdf(bins[n_underflow:], "2018", "magdown", "cf", bdt_cut=True)

    # Generate points in the whole region
    combined, sqrt_initial_guess, alt_initial_guess = _gen(
        bins, alt_bkg, (gen_low, gen_high), bkg_pdf
    )

    # Bin points
    counts, errs = stats.counts(combined, bins)

    # We don't want to fit to the stuff in the underflow bins
    sqrt_fitter = fit.binned_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        sqrt_initial_guess,
        (fit_low, gen_high),  # Fit to reduced region
    )
    alt_bkg_fitter = fit.alt_bkg_fit(
        counts[n_underflow:],
        bins[n_underflow:],
        alt_initial_guess,
        bkg_pdf,
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
        bkg_pdf,
        (fit_low, gen_high),
        alt_bkg_fitter.values,
    )

    axes["A"].set_title(
        f"Sqrt bkg fit; n_sig={sqrt_fitter.values[0]:.2f}$\pm${sqrt_fitter.errors[0]:.2f}"
    )
    axes["B"].set_title(
        f"Alt bkg fit; n_sig={alt_bkg_fitter.values[0]:.2f}$\pm${alt_bkg_fitter.errors[0]:.2f}"
    )

    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig(f"toy_mass_fit_{alt_bkg=}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy fits with the sqrt and alt bkg.")

    parser.add_argument(
        "--alt_bkg",
        help="Generate bkg distribution with the alternate background model",
        action="store_true",
    )

    main(**vars(parser.parse_args()))
