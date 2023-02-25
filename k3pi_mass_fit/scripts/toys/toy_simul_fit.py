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

from libFit import fit, toy_utils, plotting, definitions, pdfs, util
from lib_data import stats


def _gen(n_sig: int, n_bkg: int, sign: str):
    """
    Generate points

    Same signal model, slightly different bkg models

    """
    time_bin = 5

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
    n_rs_sig, n_ws_sig, n_bkg = 20_000_000, 100_000, 300_000

    rs_masses = _gen(n_rs_sig, n_bkg, "cf")
    ws_masses = _gen(n_ws_sig, n_bkg, "dcs")

    # Perform fit
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    rs_counts, rs_errs = stats.counts(rs_masses, bins)
    ws_counts, ws_errs = stats.counts(ws_masses, bins)

    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts,
        ws_counts,
        bins,
        5,
        (fit_low, gen_high),
        rs_errs,
        ws_errs,
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
