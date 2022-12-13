"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_efficiency"))

from libFit import pdfs, fit, toy_utils, plotting
from lib_efficiency.metrics import _counts


def _toy_fit():
    """
    Generate some stuff, do a fit to it

    """
    sign, time_bin = "RS", 5

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_rs_sig, n_ws_sig, n_bkg = 400_000, 20_000, 300_000
    rng = np.random.default_rng()
    rs_masses, _ = toy_utils.gen_points(rng, n_rs_sig, n_bkg, sign, time_bin)
    ws_masses, _ = toy_utils.gen_points(rng, n_ws_sig, n_bkg, sign, time_bin)

    # Perform fit
    time_bin = 5
    bins = np.linspace(*pdfs.domain(), 250)

    rs_counts, rs_errs = _counts(rs_masses, np.ones_like(rs_masses), bins)
    ws_counts, ws_errs = _counts(ws_masses, np.ones_like(ws_masses), bins)
    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts, ws_counts, bins, time_bin, rs_errs, ws_errs
    )

    fig, _ = plotting.simul_fits(
        rs_counts, rs_errs, ws_counts, ws_errs, bins, binned_fitter.values, binned=True
    )
    fig.suptitle("Binned")
    fig.tight_layout()
    fig.savefig("binned_toy_simul_fit.png")


def main():
    """
    just do 1 fit

    """
    _toy_fit()


if __name__ == "__main__":
    main()
