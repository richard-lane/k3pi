"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import argparse
import numpy as np

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import fit, toy_utils, plotting, definitions, pdfs
from lib_data import stats


def _toy_fit(alt_bkg: bool):
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
    n_rs_sig, n_ws_sig, n_bkg = 20_000_000, 100_000, 300_000
    rng = np.random.default_rng()
    rs_masses, _ = toy_utils.gen_points(
        rng, n_rs_sig, n_bkg, sign, time_bin, bkg_params, bkg_kw
    )
    ws_masses, _ = toy_utils.gen_points(
        rng, n_ws_sig, n_bkg, sign, time_bin, bkg_params, bkg_kw
    )

    # Perform fit
    bins = definitions.mass_bins(200)

    rs_counts, rs_errs = stats.counts(rs_masses, bins)
    ws_counts, ws_errs = stats.counts(ws_masses, bins)
    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts,
        ws_counts,
        bins,
        time_bin,
        rs_errs,
        ws_errs,
    )
    fig, _ = plotting.simul_fits(
        rs_counts,
        rs_errs,
        ws_counts,
        ws_errs,
        bins,
        binned_fitter.values,
    )
    model_str = "sqrt model" if not alt_bkg else "alt model"
    fig.suptitle(f"{model_str}; sqrt fit")
    fig.tight_layout()
    fig.savefig(f"simul_{model_str.replace(' ', '_')}_sqrt_fit.png")
    try:
        alt_fitter = fit.alt_simultaneous_fit(
            rs_counts,
            ws_counts,
            bins,
            time_bin,
            rs_errs,
            ws_errs,
            bdt_cut=False,
            efficiency=False,
        )

        fig, _ = plotting.alt_bkg_simul(
            rs_counts,
            rs_errs,
            ws_counts,
            ws_errs,
            bins,
            alt_fitter.values,
            bdt_cut=False,
            efficiency=False,
        )
        fig.suptitle(f"{model_str}; alt fit")
        fig.tight_layout()
        fig.savefig(f"simul_{model_str.replace(' ', '_')}_alt_fit.png")
    except FileNotFoundError:
        pass


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
