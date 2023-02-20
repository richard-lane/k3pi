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


def _toy_fit():
    """
    Generate some stuff, do a fit to it

    """
    bkg_params = pdfs.background_defaults("cf")

    # Generate points along the whole range
    (gen_low, gen_high) = pdfs.domain()

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_rs_sig, n_ws_sig, n_bkg = 20_000_000, 100_000, 300_000
    rng = np.random.default_rng()
    rs_masses, _ = toy_utils.gen_points(
        rng, n_rs_sig, n_bkg, (gen_low, gen_high), bkg_params=bkg_params, verbose=True
    )
    ws_masses, _ = toy_utils.gen_points(
        rng, n_ws_sig, n_bkg, (gen_low, gen_high), bkg_params=bkg_params, verbose=True
    )

    # Perform fit
    fit_low, _ = pdfs.reduced_domain()
    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 145.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    rs_counts, rs_errs = stats.counts(rs_masses, bins)
    ws_counts, ws_errs = stats.counts(ws_masses, bins)

    # This is the one it gets generated with, probably
    time_bin = 5
    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts,
        ws_counts,
        bins,
        time_bin,
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
    model_str = "sqrt model"
    fig.suptitle(f"{model_str}; sqrt fit")
    fig.tight_layout()

    path = f"simul_{model_str.replace(' ', '_')}_sqrt_fit.png"
    print(f"saving {path}")
    fig.savefig(path)


def main():
    """
    just do 1 fit

    """
    _toy_fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Toy simultaneous fit; sqrt model only for now"
    )

    main(**vars(parser.parse_args()))
