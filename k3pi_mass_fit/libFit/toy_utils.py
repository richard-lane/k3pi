"""
Utilities for doing a toy study

All generators use the whole domain

"""
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import pdfs


def _n_expected_normalised(
    n_gen: int,
    domain: Tuple[float, float],
    pdf_range: Tuple[float, float],
) -> float:
    """
    The number of events we expect to accept, given a number generated
    assuming the pdf is normalised

    :param n_gen: number we're generating total
    :param domain: (low, high) x
    :param range: (low, high) y

    """
    low, high = domain
    min_, max_ = pdf_range

    total_area = (max_ - min_) * (high - low)

    return n_gen / total_area


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
    low, high = pdfs.domain()

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
    verbose: bool = True,
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
    n_sig: int,
    n_bkg: int,
    pdf_domain: Tuple[float, float],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    raise NotImplementedError


def n_expected_sig(n_gen: int, signal_params: Tuple):
    """
    The average number we would expect to accept given the
    signal parameters

    """
    domain = pdfs.domain()

    return _n_expected_normalised(
        n_gen,
        domain,
        (0.0, _max(lambda pts: pdfs.normalised_signal(pts, *signal_params, domain))),
    )


def n_expected_bkg(n_gen: int, bkg_params: Tuple):
    """
    The average number we would expect to accept given the
    bkg parameters

    """
    domain = pdfs.domain()

    return _n_expected_normalised(
        n_gen,
        domain,
        (0.0, _max(lambda pts: pdfs.normalised_bkg(pts, *bkg_params, domain))),
    )
