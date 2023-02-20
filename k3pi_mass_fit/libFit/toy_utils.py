"""
Utilities for doing a toy study

"""
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from . import pdfs, bkg as lib_bkg


def _gen(
    rng: np.random.Generator,
    pdf: Callable[[np.ndarray], np.ndarray],
    n_gen: int,
    pdf_max: float,
    pdf_domain: Tuple[float, float],
    plot=False,
) -> np.ndarray:
    """
    Generate samples from a pdf

    """
    if not n_gen:
        return np.array([])

    x = pdf_domain[0] + (pdf_domain[1] - pdf_domain[0]) * rng.random(n_gen)

    y = pdf_max * rng.random(n_gen)

    f_eval = pdf(x)

    keep = y < f_eval

    if plot:
        _, axis = plt.subplots()

        pts = np.linspace(*pdf_domain, 1000)
        axis.plot(pts, pdf(pts))
        axis.scatter(x[keep], y[keep], c="k", marker=".")
        axis.scatter(x[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return x[keep]


def _max(pdf: Callable, domain: Tuple[float, float]) -> float:
    """
    Find the maximum value of a function of 1 dimension

    Then multiply it by 1.1 just to be safe

    """
    # Cut the last value off to stop it falling into
    # an overflow bin for the alt bkg model
    return 1.1 * np.max(pdf(np.linspace(*domain, 100)[:-1]))


def _gen_bkg_sqrt(
    rng: np.random.Generator,
    n_bkg: int,
    pdf_domain: Tuple[float, float],
    params: Tuple[float, float],
):
    """
    Generate background points using the sqrt model

    """
    assert len(params) == 2

    def bkg_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.normalised_bkg(x, *params, pdf_domain)

    return _gen(
        rng,
        bkg_pdf,
        n_bkg,
        _max(bkg_pdf, pdf_domain),
        pdf_domain=pdf_domain,
        plot=False,
    )


def _gen_bkg_empirical(
    rng: np.random.Generator,
    n_bkg: int,
    pdf_domain: Tuple[float, float],
    params: Tuple[float, float, float],
    *,
    n_bins: int,
    sign: str,
    bdt_cut: bool,
    efficiency: bool,
):
    """
    Generate background points using the estimated model

    """
    assert len(params) == 3

    estimated_bkg = lib_bkg.pdf(n_bins, sign, bdt_cut=bdt_cut, efficiency=efficiency)

    def bkg_pdf(x):
        return pdfs.estimated_bkg(x, estimated_bkg, *params)

    return _gen(
        rng,
        bkg_pdf,
        n_bkg,
        _max(bkg_pdf, pdf_domain),
        pdf_domain=pdf_domain,
        plot=False,
    )


def _gen_sig(
    rng: np.random.Generator,
    n_sig: int,
    pdf_domain: Tuple[float, float],
    params: Tuple,
) -> np.ndarray:
    """
    Generate signal points

    """
    centre, width, alpha, beta = params

    def signal_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.normalised_signal(
            x, centre, width, width, alpha, alpha, beta, pdf_domain
        )

        # Removed because it barely makes it faster
        # Choose regions for the generation
        # low, high = pdfs.domain()
        # a, b = 144.0, 148.0
        # h = 0.1

        # area_low = h * (a - low)
        # area_mid = b - a
        # area_high = h * (high - b)

        # # num to generate low + high is based on their areas
        # n_low = int(n_sig * area_low / area_mid)
        # n_high = int(n_sig * area_high / area_mid)

        # # Generate
        # x_low = low + (a - low) * rng.random(n_low)
        # x_mid = a + (b - a) * rng.random(n_sig)
        # x_high = b + (high - b) * rng.random(n_high)

        # y_low = 0.1 * rng.random(n_low)
        # y_mid = rng.random(n_sig)
        # y_high = 0.1 * rng.random(n_high)

        # return np.concatenate(
        #     (
        #         x_low[y_low < signal_pdf(x_low)],
        #         x_mid[y_mid < signal_pdf(x_mid)],
        #         x_high[y_high < signal_pdf(x_high)],
        #     )
        # )

    return _gen(
        rng,
        signal_pdf,
        n_sig,
        _max(signal_pdf, pdf_domain),
        pdf_domain,
        plot=False,
    )


def gen_points(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    pdf_domain: Tuple[float, float],
    *,
    bkg_params: Tuple[float, float] = (0, 0),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sig and n_bkg points; see which are kept using accept-reject; return an array of both

    Generates bkg using the sqrt model

    Also returns true fit params (n_sig, n_bkg, centre, width, alpha, beta, *bkg_params)

    """
    bkg = _gen_bkg_sqrt(rng, int(n_bkg), pdf_domain, bkg_params)
    if verbose:
        print(f"{bkg_params=}")
        print(f"{len(bkg)=}")

    centre, width, alpha, beta = pdfs.signal_defaults(time_bin=5)
    sig = _gen_sig(
        rng,
        int(n_sig),
        pdf_domain,
        (centre, width, alpha, beta),
    )
    if verbose:
        print(f"{len(sig)=}")

    true_params = np.array(
        (len(sig), len(bkg), centre, width, width, alpha, alpha, beta, *bkg_params)
    )

    return np.concatenate((sig, bkg)), true_params


def gen_points_alt_bkg(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    pdf_domain: Tuple[float, float],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    raise NotImplementedError
