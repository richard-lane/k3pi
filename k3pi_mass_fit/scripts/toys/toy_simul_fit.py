"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import argparse
import numpy as np
from typing import Tuple

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import fit, toy_utils, plotting, definitions, pdfs, util
from lib_data import stats


def _gen(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    fit_region: Tuple[float, float],
):
    """
    Generate points

    Same signal model, slightly different bkg models

    returns true params also

    """
    time_bin = 5

    sig_params = util.signal_param_guess(time_bin)
    bkg_params = util.sqrt_bkg_param_guess(sign)

    sig = toy_utils.gen_sig(rng, n_sig, sig_params, verbose=True)
    bkg = toy_utils.gen_bkg_sqrt(rng, n_bkg, bkg_params, verbose=True)

    # Number we expect to generate
    n_sig = toy_utils.n_expected_sig(n_sig, fit_region, sig_params)
    n_bkg = toy_utils.n_expected_bkg(n_bkg, fit_region, bkg_params)

    return np.concatenate((sig, bkg)), (n_sig, n_bkg, *sig_params, *bkg_params)


def main():
    """
    just do 1 fit

    """
    n_rs_sig, n_ws_sig, n_bkg = 20_000_000, 100_000, 300_000

    rng = np.random.default_rng()

    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    fit_region = (fit_low, gen_high)
    rs_masses, rs_params = _gen(rng, n_rs_sig, n_bkg, "cf", fit_region)
    ws_masses, ws_params = _gen(rng, n_ws_sig, n_bkg, "dcs", fit_region)

    # Perform fit
    initial_guess = util.fit_params(rs_params, ws_params)

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    rs_counts, rs_errs = stats.counts(rs_masses, bins)
    ws_counts, ws_errs = stats.counts(ws_masses, bins)

    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts[n_underflow:],
        ws_counts[n_underflow:],
        bins[n_underflow:],
        initial_guess,
        (fit_low, gen_high),
        rs_errors=rs_errs[n_underflow:],
        ws_errors=ws_errs[n_underflow:],
    )

    fig, _ = plotting.simul_fits(
        rs_counts,
        rs_errs,
        ws_counts,
        ws_errs,
        bins,
        (fit_low, gen_high),
        binned_fitter.values,
    )
    fig.suptitle("sqrt model; sqrt fit")
    fig.tight_layout()

    path = "simul_sqrt_model_sqrt_fit.png"
    print(f"saving {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Toy simultaneous fit; sqrt model only for now"
    )

    main(**vars(parser.parse_args()))
