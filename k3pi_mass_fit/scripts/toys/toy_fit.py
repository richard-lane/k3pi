"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

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
    n_sig, n_bkg = 1600000, 800000
    combined, true_params = toy_utils.gen_points(
        np.random.default_rng(), n_sig, n_bkg, sign, time_bin
    )

    # Perform fit
    sig_frac = true_params[0]
    bins = np.linspace(*pdfs.domain(), 500)
    counts, errs = _counts(combined, np.ones_like(combined), bins)
    binned_fitter = fit.binned_fit(counts, bins, sign, time_bin, sig_frac)

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nCCC")
    plotting.mass_fit(
        (axes["A"], axes["C"]), counts, errs, bins, binned_fitter.values
    )

    axes["A"].set_title("Binned")

    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig("toy_mass_fit.png")


def main():
    """
    just do 1 fit

    """
    _toy_fit()


if __name__ == "__main__":
    main()
