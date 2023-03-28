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

from libFit import pdfs, fit, toy_utils, plotting, definitions, util, bkg as lib_bkg
from lib_data import stats


def _gen(
    bins: np.ndarray, alt_bkg: bool, domain: Tuple[float, float]
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
    sig = toy_utils.gen_sig(rng, n_sig, sig_params, pdfs.reduced_domain(), verbose=True)
    if not alt_bkg:
        bkg = toy_utils.gen_bkg_sqrt(
            rng, n_bkg, sqrt_bkg_params, pdfs.reduced_domain(), verbose=True
        )
    else:
        # Hard coded for now
        bkg_pdf = lib_bkg.pdf(bins, "2018", "magdown", "dcs", bdt_cut=False)
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
    # Perform fits to the restricted range
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    combined, sqrt_initial_guess, alt_initial_guess = _gen(
        bins, alt_bkg, (fit_low, gen_high)
    )

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
        "2018",
        "magdown",
        "dcs",
        alt_initial_guess,
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

    plt.savefig(f"toy_mass_fit_{alt_bkg=}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy fits with the sqrt and alt bkg.")

    parser.add_argument(
        "--alt_bkg",
        help="Generate bkg distribution with the alternate background model",
        action="store_true",
    )

    main(**vars(parser.parse_args()))
