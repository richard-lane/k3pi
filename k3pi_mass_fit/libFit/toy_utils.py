"""
Utilities for doing a toy study

All generators use the whole domain

"""
import sys
import pathlib
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import pdfs, bkg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
from lib_data import stats


def _n_expected(
    n_gen: int,
    domain: Tuple[float, float],
    pdf_range: Tuple[float, float],
    fit_region_integral: float,
) -> float:
    """
    The number of events we expect to accept in the fit region,
    given a number generated

    :param n_gen: number we're generating total
    :param domain: (low, high) x for generating
    :param range: (low, high) y
    :param fit_region_integral: the integral of the PDF in the fit region

    """
    # Total number accepted in the fit region is
    #   (n gen tot * n_gen in fit region * integral in fit region) / (height * fit region width)
    # The width of the fit region cancels out here if we write
    #   (n_gen in fit region) = (fit region width / domain width)
    # Giving
    #   (n gen tot * integral in fit region * fit region width) / (height * domain width)
    low, high = domain
    min_, max_ = pdf_range

    domain_width = high - low
    range_height = max_ - min_

    return n_gen * fit_region_integral / (domain_width * range_height)


def _max(pdf: Callable) -> float:
    """
    Find the maximum value of a function of 1 dimension

    Then multiply it by 1.1 just to be safe

    """
    # Cut the last value off to stop it falling into
    # an overflow bin for the alt bkg model
    return 1.1 * np.max(pdf(np.linspace(*pdfs.domain(), 100)[:-1]))


def _gen(
    rng: np.random.Generator,
    pdf: Callable[[np.ndarray], np.ndarray],
    domain: Tuple,
    n_gen: int,
    plot=False,
) -> np.ndarray:
    """
    Generate samples from a pdf

    also returns max

    """
    if not n_gen:
        return np.array([])

    pdf_max = _max(pdf)
    low, high = domain

    x = low + (high - low) * rng.random(n_gen)

    y = pdf_max * rng.random(n_gen)

    f_eval = pdf(x)
    assert (f_eval < pdf_max).all()

    keep = y < f_eval

    if plot:
        _, axis = plt.subplots()

        pts = np.linspace(*pdfs.domain(), 1000)
        axis.plot(pts, pdf(pts))
        axis.scatter(x[keep], y[keep], c="k", marker=".")
        axis.scatter(x[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return x[keep]


def gen_bkg_sqrt(
    rng: np.random.Generator,
    n_gen: int,
    bkg_params: Tuple,
    *,
    verbose: bool = False,
):
    """
    Generate points according to the signal model - quite slow and inefficient

    Generates along the whole of pdfs.domain()

    :param rng: random number generator
    :param n_gen: number to generate
    :param bkg_params: parameters to use in the generation
    :param verbose: whether to print stuff

    :returns: array of points

    """
    assert len(bkg_params) == 2, "Wrong number bkg params"

    points = _gen(
        rng,
        lambda pts: pdfs.normalised_bkg(pts, *bkg_params, pdfs.domain()),
        pdfs.domain(),
        n_gen,
        plot=False,
    )

    if verbose:
        print(
            f"bkg generated: {len(points)}; efficiency {100 * len(points) / n_gen:.2f}%"
        )

    return points


def gen_sig(
    rng: np.random.Generator,
    n_gen: int,
    sig_params: Tuple,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points according to the signal model - quite slow and inefficient

    Generates along the whole of pdfs.domain()

    :param rng: random number generator
    :param n_gen: number to generate
    :param sig_params: parameters to use in the generation
    :param verbose: whether to print stuff

    :returns: array of points

    """
    assert len(sig_params) == 6, "Wrong number sig params"

    points = _gen(
        rng,
        lambda pts: pdfs.normalised_signal(pts, *sig_params, pdfs.domain()),
        pdfs.domain(),
        n_gen,
        plot=False,
    )

    if verbose:
        print(
            f"signal generated: {len(points)}; efficiency {100 * len(points) / n_gen:.2f}%"
        )

    return points


def gen_alt_bkg(
    rng: np.random.Generator,
    n_gen: int,
    bkg_pdf: Callable,
    bkg_params: Tuple,
    pdf_domain: Tuple,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points according to the alternate background model

    :param rng: rng
    :param n_gen: number to generate
    :param bkg_pdf: a PDF like the one returned from
                    bkg.pdf
    :param bkg_params: 3 params for alt bkg model
    :param pdf_domain: domain along which the provided PDF is normalised
    :param verbose: whether to print stuff

    """
    assert len(bkg_params) == 3, "Wrong number alt bkg params"

    points = _gen(
        rng,
        lambda pts: pdfs.estimated_bkg(pts, bkg_pdf, pdf_domain, *bkg_params),
        pdf_domain,
        n_gen,
        plot=False,
    )

    if verbose:
        print(
            f"bkg generated: {len(points)}; efficiency {100 * len(points) / n_gen:.2f}%"
        )

    return points


def n_expected_sig(n_gen: int, domain: Tuple[float, float], signal_params: Tuple):
    """
    The average number we would expect to accept given the
    signal parameters

    """
    # Find the integral of the PDF in domain
    # The PDF is normalised along the whole of pdfs.domain()
    pts = np.linspace(*domain, 100_000)
    integral = stats.integral(
        pts, pdfs.normalised_signal(pts, *signal_params, pdfs.domain())
    )

    return _n_expected(
        n_gen,
        domain,
        (
            0.0,
            _max(
                lambda pts: pdfs.normalised_signal(pts, *signal_params, pdfs.domain())
            ),
        ),
        integral,
    )


def n_expected_bkg(n_gen: int, domain: Tuple[float, float], bkg_params: Tuple):
    """
    The average number we would expect to accept given the
    bkg parameters

    """
    # Find the integral of the PDF in domain
    # The PDF is normalised along the whole of pdfs.domain()
    pts = np.linspace(*domain, 100_000)
    integral = stats.integral(pts, pdfs.normalised_bkg(pts, *bkg_params, pdfs.domain()))

    return _n_expected(
        n_gen,
        domain,
        (0.0, _max(lambda pts: pdfs.normalised_bkg(pts, *bkg_params, pdfs.domain()))),
        integral,
    )


def n_expected_alt_bkg(
    n_gen: int,
    domain: Tuple[float, float],
    bkg_params: Tuple,
    bins: np.ndarray,
    year,
    sign,
    magnetisation,
    *,
    bdt_cut,
):
    """
    The average number we would expect to accept given the
    bkg parameters

    :param n_gen: total number generated before acc-rej
    :param domain: fit region
    :param bkg_params: 3 params for bkg fit

    :param bins: for finding the right pdf, should be defined over the whole generator domain
    :param year: for finding the right pdf
    :param sign: for finding the right pdf
    :param magnetisation: for finding the right pdf
    :param bdt_cut: for finding the right pdf

    """
    pdf = bkg.pdf(bins, year, magnetisation, sign, bdt_cut=bdt_cut)

    # Find the integral of the PDF in domain
    # The PDF is normalised along the whole of pdfs.domain()
    pts = np.linspace(*domain, 1_000_000)[
        :-1
    ]  # Cut the last point off so we don't overflow
    y_vals = pdfs.estimated_bkg(pts, pdf, pdfs.domain(), *bkg_params)

    integral = stats.integral(pts, y_vals)

    return _n_expected(
        n_gen,
        domain,
        (
            0.0,
            _max(lambda pts: pdfs.estimated_bkg(pts, pdf, pdfs.domain(), *bkg_params)),
        ),
        integral,
    )
