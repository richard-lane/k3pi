"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_efficiency"))

from libFit import pdfs, fit, toy_utils
from lib_efficiency.metrics import _counts


def _plot(axis: plt.Axes, params: Tuple, fmt: str, label: str) -> None:
    """Plot fit result on an axis"""
    pts = np.linspace(*pdfs.domain(), 250)
    axis.plot(
        pts,
        pdfs.fractional_pdf(pts, *params),
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
    n_rs_sig, n_ws_sig, n_bkg = 400_000, 20000, 300_000
    rng = np.random.default_rng()
    rs_masses, _ = toy_utils.gen_points(rng, n_rs_sig, n_bkg, sign, time_bin)
    ws_masses, _ = toy_utils.gen_points(rng, n_ws_sig, n_bkg, sign, time_bin)

    # Perform fit
    time_bin = 5
    bins = np.linspace(*pdfs.domain(), 250)
    unbinned_fitter = fit.simultaneous_fit(rs_masses, ws_masses, time_bin)
    rs_counts, rs_errs = _counts(rs_masses, np.ones_like(rs_masses), bins)
    ws_counts, ws_errs = _counts(ws_masses, np.ones_like(ws_masses), bins)
    binned_fitter = fit.binned_simultaneous_fit(
        rs_counts, ws_counts, bins, time_bin, rs_errs, ws_errs
    )
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(rs_masses, bins=250, density=True)
    axes[1].hist(ws_masses, bins=250, density=True)

    _plot(axes[0], unbinned_fitter.values[:-1], "r--", "Unbinned")
    _plot(
        axes[1],
        (unbinned_fitter.values[-1], *unbinned_fitter.values[1:-1]),
        "r--",
        "Unbinned",
    )

    _plot(
        axes[0], (binned_fitter.values[0], *binned_fitter.values[2:]), "k--", "Binned"
    )
    _plot(
        axes[1],
        binned_fitter.values[1:],
        "k--",
        "Binned",
    )

    axes[0].legend()
    for axis in axes:
        axis.set_xlabel(r"$\Delta M$ /MeV")
    fig.suptitle("toy data")
    fig.tight_layout()

    plt.savefig("toy_simul_fit.png")


def main():
    """
    just do 1 fit

    """
    _toy_fit()


if __name__ == "__main__":
    main()
