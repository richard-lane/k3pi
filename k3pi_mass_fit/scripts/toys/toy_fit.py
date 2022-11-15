"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_efficiency"))

from libFit import pdfs, fit, toy_utils
from lib_efficiency.metrics import _counts


def _plot(axis: plt.Axes, fitter: Minuit, fmt: str, label: str) -> None:
    """Plot fit result on an axis"""
    pts = np.linspace(*pdfs.domain(), 250)
    axis.plot(
        pts,
        pdfs.fractional_pdf(pts, *fitter.values),
        fmt,
        label=label,
    )


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
    unbinned_fitter = fit.fit(combined, sign, time_bin, sig_frac)
    bins = np.linspace(*pdfs.domain(), 500)
    counts, errs = _counts(combined, np.ones_like(combined), bins)
    binned_fitter = fit.binned_fit(counts, bins, sign, time_bin, sig_frac)

    fig, axis = plt.subplots()
    axis.hist(combined, bins=250, density=True)

    _plot(axis, unbinned_fitter, "r--", "Unbinned")
    _plot(axis, binned_fitter, "k--", "Binned")

    axis.legend()
    axis.set_xlabel(r"$\Delta M$ /MeV")
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
