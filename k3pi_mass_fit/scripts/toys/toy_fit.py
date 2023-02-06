"""
from multiprocessing import Process, Manager
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


def _toy_fit(alt_bkg):
    """
    Generate some stuff, do a fit to it

    """
    sign, time_bin = "cf", 5

    bkg_params = (0, 0, 0) if alt_bkg else pdfs.background_defaults(sign)
    bkg_kw = (
        {"n_bins": 100, "sign": sign, "bdt_cut": False, "efficiency": False}
        if alt_bkg
        else {}
    )

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 2_000_000, 5_000_000
    combined, true_params = toy_utils.gen_points(
        np.random.default_rng(),
        n_sig,
        n_bkg,
        sign,
        time_bin,
        bkg_params,
        bkg_kw,
    )

    # Perform fits
    sig_frac = true_params[0]
    bins = definitions.mass_bins(100)
    counts, errs = stats.counts(combined, bins)
    binned_fitter = fit.binned_fit(counts, bins, sign, time_bin, sig_frac)

    fig, axes = plt.subplot_mosaic("AAABBB\nAAABBB\nAAABBB\nCCCDDD", figsize=(12, 8))
    plotting.mass_fit((axes["A"], axes["C"]), counts, errs, bins, binned_fitter.values)

    try:
        alt_fitter = fit.alt_bkg_fit(
            counts, bins, sign, time_bin, sig_frac, bdt_cut=False, efficiency=False
        )
        plotting.alt_bkg_fit(
            (axes["B"], axes["D"]),
            counts,
            errs,
            bins,
            alt_fitter.values,
            sign=sign,
            bdt_cut=False,
            efficiency=False,
        )
        axes["B"].set_title("Alt bkg fit")
    except FileNotFoundError:
        axes["B"].set_title("Alt bkg not possible; pickle dump not created")

    axes["A"].set_title("Sqrt bkg fit")

    fig.suptitle(f"toy data{' alt bkg' if alt_bkg else ''}")
    fig.tight_layout()

    plt.savefig(f"toy_mass_fit{'_altbkg' if alt_bkg else ''}.png")


def main(args):
    """
    just do 1 fit

    """
    alt_bkg = args.alt_bkg
    _toy_fit(alt_bkg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt_bkg", action="store_true")

    main(parser.parse_args())
