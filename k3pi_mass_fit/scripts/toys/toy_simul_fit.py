"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import argparse
from typing import Callable

import numpy as np

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import fit, toy_utils, plotting, definitions, pdfs, util, bkg as lib_bkg
from lib_data import stats


def _gen(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    bkg_pdf: Callable = None,
):
    """
    Generate points in the fit region

    Same signal model, slightly different bkg models

    """
    # Generate signal
    sig = toy_utils.gen_sig(
        rng, n_sig, util.signal_param_guess(5), pdfs.reduced_domain(), verbose=True
    )

    # Generate bkg
    if bkg_pdf is None:
        # Just use cf bkg for both, its fine
        bkg = toy_utils.gen_bkg_sqrt(
            rng,
            n_bkg,
            util.sqrt_bkg_param_guess("cf"),
            pdfs.reduced_domain(),
            verbose=True,
        )

    else:
        bkg = toy_utils.gen_alt_bkg(
            rng, n_bkg, bkg_pdf, (0, 0, 0), pdfs.reduced_domain(), verbose=True
        )

    return np.concatenate((sig, bkg))


def main(*, alt_fit: bool, alt_gen: bool):
    """
    just do 1 fit

    """
    n_rs_sig, n_ws_sig, n_bkg = 800_000, 2_200, 30_000

    rng = np.random.default_rng()

    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, gen_high), (n_underflow, 150)
    )

    # Will need this for fitting/plotting/generating
    if alt_gen or alt_fit:
        bkg_pdf = lib_bkg.pdf(bins[n_underflow:], "2018", "magdown", "cf", bdt_cut=True)

    # Bkg_pdf = None will generate with sqrt model
    rs_masses = _gen(rng, n_rs_sig, n_bkg, bkg_pdf if alt_gen else None)
    ws_masses = _gen(rng, n_ws_sig, n_bkg, bkg_pdf if alt_gen else None)

    # Perform fit
    initial_guess = (
        (n_rs_sig, n_bkg, n_ws_sig, n_bkg, *util.signal_param_guess(5), 0, 0, 0, 0)
        if not alt_fit
        else (
            n_rs_sig,
            n_bkg,
            n_ws_sig,
            n_bkg,
            *util.signal_param_guess(5),
            0,
            0,
            0,
            0,
            0,
            0,
        )
    )
    rs_counts, rs_errs = stats.counts(rs_masses, bins)
    ws_counts, ws_errs = stats.counts(ws_masses, bins)

    binned_fitter = (
        fit.binned_simultaneous_fit(
            rs_counts[n_underflow:],
            ws_counts[n_underflow:],
            bins[n_underflow:],
            initial_guess,
            (fit_low, gen_high),
            rs_errors=rs_errs[n_underflow:],
            ws_errors=ws_errs[n_underflow:],
        )
        if not alt_fit
        else fit.alt_simultaneous_fit(
            rs_counts[n_underflow:],
            ws_counts[n_underflow:],
            bins[n_underflow:],
            initial_guess,
            bkg_pdf,
            bkg_pdf,
            rs_errors=rs_errs[n_underflow:],
            ws_errors=ws_errs[n_underflow:],
        )
    )

    fig, axes = (
        plotting.simul_fits(
            rs_counts,
            rs_errs,
            ws_counts,
            ws_errs,
            bins,
            (fit_low, gen_high),
            binned_fitter.values,
        )
        if not alt_fit
        else plotting.alt_bkg_simul(
            rs_counts,
            rs_errs,
            ws_counts,
            ws_errs,
            bins,
            (fit_low, gen_high),
            bkg_pdf,
            bkg_pdf,
            binned_fitter.values,
        )
    )
    axes["A"].set_title(
        f"n_sig={binned_fitter.values[0]:,.2f}$\pm${binned_fitter.errors[0]:,.2f}"
    )
    axes["B"].set_title(
        f"n_sig={binned_fitter.values[2]:,.2f}$\pm${binned_fitter.errors[2]:,.2f}"
    )

    model_str = "sqrt model" if not alt_gen else "alt model"
    fit_str = "sqrt fit" if not alt_fit else "alt fit"
    fig.suptitle(f"{model_str}; {fit_str}")
    fig.tight_layout()

    path = f"simul_fit_{model_str.replace(' ', '_')}_{fit_str.replace(' ', '_')}.png"
    print(f"saving {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy simultaneous fit")

    parser.add_argument(
        "--alt_fit",
        help="Fit with the alternate background model",
        action="store_true",
    )
    parser.add_argument(
        "--alt_gen",
        help="Generate bkg distribution with the alternate background model",
        action="store_true",
    )

    main(**vars(parser.parse_args()))
