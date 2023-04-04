"""
Utilities for doing a toy study

All generators use the whole domain

"""
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import pdfs


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

    retval = np.empty(n_gen) * np.nan
    num_generated = 0

    # Generate in chunks since we don't know how many will be accepted
    chunk_size = 2 * n_gen
    while num_generated < n_gen:
        uniform_x = low + (high - low) * rng.random(chunk_size)
        uniform_y = pdf_max * rng.random(chunk_size)

        f_eval = pdf(uniform_x)
        assert (f_eval < pdf_max).all()

        keep = uniform_y < f_eval
        n_in_chunk = np.sum(keep)

        # Fill in with numpy array indexing
        try:
            retval[num_generated : num_generated + n_in_chunk] = uniform_x[keep]
            num_generated += n_in_chunk

        except ValueError:
            # Unless we go over the end, in which case just fill in the ones we need
            n_left = n_gen - np.sum(~np.isnan(retval))
            retval[num_generated:] = uniform_x[keep][:n_left]
            break

    if plot:
        _, axis = plt.subplots()

        pts = np.linspace(*pdfs.domain(), 1000)
        axis.plot(pts, pdf(pts))
        axis.scatter(uniform_x[keep], uniform_y[keep], c="k", marker=".")
        axis.scatter(uniform_x[~keep], uniform_y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return retval


def gen_bkg_sqrt(
    rng: np.random.Generator,
    n_gen: int,
    bkg_params: Tuple,
    domain: Tuple[float, float],
    *,
    verbose: bool = False,
):
    """
    Generate points according to the signal model - quite slow and inefficient

    Generates along the whole of pdfs.domain()

    :param rng: random number generator
    :param n_gen: number to generate
    :param bkg_params: parameters to use in the generation
    :param domain: domain to generate along
    :param verbose: whether to print stuff

    :returns: array of points

    """
    assert len(bkg_params) == 2, "Wrong number bkg params"

    points = _gen(
        rng,
        lambda pts: pdfs.normalised_bkg(pts, *bkg_params, pdfs.domain()),
        domain,
        n_gen,
        plot=False,
    )

    if verbose:
        print(f"bkg generated: {len(points)}")

    return points


def gen_sig(
    rng: np.random.Generator,
    n_gen: int,
    sig_params: Tuple,
    domain: Tuple[float, float],
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points according to the signal model - quite slow and inefficient

    Generates along the whole of pdfs.domain()

    :param rng: random number generator
    :param n_gen: number to generate
    :param sig_params: parameters to use in the generation
    :param domain: domain to generate along
    :param verbose: whether to print stuff

    :returns: array of points

    """
    assert len(sig_params) == 6, "Wrong number sig params"

    points = _gen(
        rng,
        lambda pts: pdfs.normalised_signal(pts, *sig_params, pdfs.domain()),
        domain,
        n_gen,
        plot=False,
    )

    if verbose:
        print(f"signal generated: {len(points)}")

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
        print(f"bkg generated: {len(points)}")

    return points
