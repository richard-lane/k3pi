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
    plot=False,
    pdf_domain=pdfs.domain(),
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
    # Cut the last value off to stop it falling into
    # an overflow bin for the alt bkg model
    return 1.1 * np.max(pdf(np.linspace(*domain, 100)[:-1]))


def _gen_bkg(
    rng: np.random.Generator,
    n_bkg: int,
    params: Tuple = None,
    *,
    n_bins: int = None,
    sign: str = None,
    bdt_cut: bool = False,
    efficiency: bool = False,
    verbose: bool = True,
    reduced: bool = False,
) -> np.ndarray:
    """
    Generate background points; returns an array

    """
    if n_bins is not None:
        gen_kw = {}
        if verbose:
            print("Generating toy points with the estimated bkg model")
        assert len(params) == 3

        estimated_bkg = lib_bkg.pdf(
            n_bins, sign, bdt_cut=bdt_cut, efficiency=efficiency
        )

        def bkg_pdf(x):
            return pdfs.estimated_bkg(x, estimated_bkg, *params)

    else:
        # Using the sqrt model thing
        if verbose:
            print("Generating toy points with the sqrt bkg model")

        assert len(params) == 2

        # These shouldn't be specified
        assert bdt_cut is False
        assert efficiency is False
        assert n_bins is None
        assert sign is None

        if reduced:

            def bkg_pdf(x: np.ndarray) -> np.ndarray:
                return pdfs.normalised_bkg_reduced(x, *params)

            gen_kw = {"pdf_domain": pdfs.reduced_domain()}

        else:

            def bkg_pdf(x: np.ndarray) -> np.ndarray:
                return pdfs.background(x, *params)

            gen_kw = {}

    return _gen(
        rng,
        bkg_pdf,
        n_bkg,
        _max(bkg_pdf, pdfs.domain()),
        plot=False,
        **gen_kw,
    )


def _gen_sig(
    rng: np.random.Generator, n_sig: int, params: Tuple, *, reduced: bool = False
) -> np.ndarray:
    """
    Generate signal points

    """
    centre, width, alpha, beta = params

    if reduced:

        def signal_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.normalised_signal_reduced(
                x, centre, width, width, alpha, alpha, beta
            )

        gen_kw = {"pdf_domain": pdfs.reduced_domain()}

    else:

        def signal_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.signal(x, centre, width, width, alpha, alpha, beta)

        gen_kw = {}

    return _gen(
        rng,
        signal_pdf,
        n_sig,
        _max(signal_pdf, pdfs.domain()),
        plot=False,
        **gen_kw,
    )


def gen_points(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    time_bin: int,
    bkg_params: Tuple,
    bkg_kw: dict = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sig and n_bkg points; see which are kept using accept-reject; return an array of both

    bkg_kw is a dict of args to pass to gen_bkg fcn,
    in case we want to pass info like n_bins for finding the
    estimated bkg fcn from a pickle dump

    Also returns true fit params (n_sig, n_bkg, centre, width, alpha, beta, *bkg_params)

    """
    bkg_kw = {} if bkg_kw is None else bkg_kw
    bkg = _gen_bkg(rng, int(n_bkg), bkg_params, **bkg_kw, verbose=verbose)

    centre, width, alpha, beta = pdfs.signal_defaults(time_bin)
    sig = _gen_sig(
        rng,
        int(n_sig),
        (centre, width, alpha, beta),
    )
    if verbose:
        print(f"{len(sig)=}\t{len(bkg)=}")

    return np.concatenate((sig, bkg)), np.array(
        (len(sig), len(bkg), centre, width, width, alpha, alpha, beta, *bkg_params)
    )


def gen_points_reduced(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    time_bin: int,
    bkg_params: Tuple,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sig and n_bkg points; see which are kept using accept-reject; return an array of both

    Also returns true fit params (n_sig, n_bkg, centre, width, alpha, beta, *bkg_params)

    """
    bkg = _gen_bkg(rng, int(n_bkg), bkg_params, verbose=verbose, reduced=True)

    centre, width, alpha, beta = pdfs.signal_defaults(time_bin)
    sig = _gen_sig(
        rng,
        int(n_sig),
        (centre, width, alpha, beta),
        reduced=True,
    )
    if verbose:
        print(f"{len(sig)=}\t{len(bkg)=}")

    return np.concatenate((sig, bkg)), np.array(
        (len(sig), len(bkg), centre, width, width, alpha, alpha, beta, *bkg_params)
    )
