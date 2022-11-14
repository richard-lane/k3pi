"""
from multiprocessing import Process, Manager
Generate some points with accept/reject, fit to them and show the fit

"""
import sys
import pathlib
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from libFit import pdfs
from libFit import fit


def _gen(
    rng: np.random.Generator,
    pdf: Callable[[np.ndarray], np.ndarray],
    n_gen: int,
    pdf_max: float,
    plot=False,
) -> np.ndarray:
    """
    Generate samples from a pdf

    """
    if not n_gen:
        return np.array([])

    pdf_domain = pdfs.domain()
    x = pdf_domain[0] + (pdf_domain[1] - pdf_domain[0]) * rng.random(n_gen)

    y = pdf_max * rng.random(n_gen)

    f_eval = pdf(x)

    keep = y < f_eval

    if plot:
        _, ax = plt.subplots()

        pts = np.linspace(*pdf_domain, 1000)
        ax.plot(pts, pdf(pts))
        ax.scatter(x[keep], y[keep], c="k", marker=".")
        ax.scatter(x[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return x[keep]


def _max(pdf: Callable, domain: Tuple[float, float]) -> float:
    """
    Find the maximum value of a function of 1 dimension

    Then multiply it by 1.1 just to be safe

    """
    return 1.1 * np.max(pdf(np.linspace(*domain, 100)))


def _noise(gen: np.random.Generator, *args: float) -> Tuple[float, ...]:
    """
    Add some random noise to some floats

    """
    noise = 0.001
    return tuple(
        noise * val
        for noise, val in zip(
            [(1 - noise) + 2 * noise * gen.random() for _ in args], args
        )
    )


def _gen_points(
    rng: np.random.Generator, n_sig: int, n_bkg: int, sign: str, time_bin: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sig and n_bkg points; see which are kept using accept-reject; return an array of both

    Also returns true fit params (signal frac, centre, width, alpha, beta, a, b)

    """
    centre, width, alpha, beta, a, b = pdfs.defaults(sign, time_bin)
    n_sig, n_bkg, centre, width_l, width_r, alpha_l, alpha_r, a, b = _noise(
        rng, float(n_sig), float(n_bkg), centre, width, width, alpha, alpha, a, b
    )

    def signal_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.signal(x, centre, width_l, width_r, alpha_l, alpha_r, beta)

    sig = _gen(
        rng,
        signal_pdf,
        int(n_sig),
        _max(signal_pdf, pdfs.domain()),
        plot=False,
    )

    def bkg_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.background(x, a, b)

    bkg = _gen(
        rng,
        bkg_pdf,
        int(n_bkg),
        _max(bkg_pdf, pdfs.domain()),
        plot=False,
    )

    return np.concatenate((sig, bkg)), np.array(
        (
            len(sig) / (len(sig) + len(bkg)),
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            a,
            b,
        )
    )


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
    combined, true_params = _gen_points(
        np.random.default_rng(), n_sig, n_bkg, sign, time_bin
    )

    # Perform fit
    sig_frac = true_params[0]
    unbinned_fitter = fit.fit(combined, sign, time_bin, sig_frac)
    binned_fitter = fit.binned_fit(
        combined, np.linspace(*pdfs.domain(), 500), sign, time_bin, sig_frac
    )

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
