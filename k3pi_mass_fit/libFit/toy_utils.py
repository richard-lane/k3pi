"""
Utilities for doing a toy study

"""
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from . import pdfs


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


def gen_points(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    time_bin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sig and n_bkg points; see which are kept using accept-reject; return an array of both

    Also returns true fit params (signal frac, centre, width, alpha, beta, a, b)

    """
    centre, width, alpha, beta, a, b = pdfs.defaults(sign, time_bin)

    def signal_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.signal(x, centre, width, width, alpha, alpha, beta)

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

    # Find signal and bkg fractions
    sig_frac = len(sig) / (len(sig) + len(bkg))
    bkg_frac = len(bkg) / (len(sig) + len(bkg))

    return np.concatenate((sig, bkg)), np.array(
        (
            sig_frac,
            bkg_frac,
            centre,
            width,
            width,
            alpha,
            alpha,
            beta,
            a,
            b,
        )
    )
