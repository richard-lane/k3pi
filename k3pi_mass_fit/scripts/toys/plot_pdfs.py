"""
Plot the PDFs which we fit the mass distribution to

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))

from libFit import pdfs, definitions, util, bkg


def _bkg(
    points: np.ndarray,
    points2: np.ndarray,
    alt_bkg: bool,
    year: str,
    sign: str,
    magnetisation: str,
    bdt_cut: bool,
) -> np.ndarray:
    """
    Background evaluated at some points

    """
    if not alt_bkg:
        bkg_params = util.sqrt_bkg_param_guess("cf")
        return (
            pdfs.normalised_bkg(points, *bkg_params, pdfs.domain()),
            pdfs.normalised_bkg(points2, *bkg_params, pdfs.reduced_domain()),
            bkg_params,
        )

    # TODO plot the actual pdf
    bins = np.linspace(*pdfs.domain(), 100)
    bins2 = np.linspace(*pdfs.reduced_domain(), 100)

    pdf = bkg.pdf(bins, year, magnetisation, sign, bdt_cut=bdt_cut)
    pdf2 = bkg.pdf(bins2, year, magnetisation, sign, bdt_cut=bdt_cut)

    vals = pdf(points)
    vals2 = pdf2(points2)

    return vals, vals2, None

    return pdfs.estimated_bkg(points, *bkg_params), bkg_params


def _pdf(points: np.ndarray, alt_bkg: bool, params: Tuple) -> np.ndarray:
    """
    Combined sig + bkg pdf evaluated at some points

    """
    n_sig = 0.2
    n_bkg = 1 - n_sig

    if not alt_bkg:
        return pdfs.model(points, n_sig, n_bkg, *params, pdfs.domain())

    # TODO the actual PDF
    return np.zeros_like(points)


def main(
    *,
    alt_bkg: bool,
    year: str = None,
    magnetisation: str = None,
    sign: str = None,
    bdt_cut: bool = None
):
    """Make and show a plot"""
    if alt_bkg:
        assert year is not None
        assert magnetisation is not None
        assert sign is not None
        assert bdt_cut is not None

    centres = np.linspace(*pdfs.domain(), 1000)[:-1]
    reduced_centres = np.linspace(*pdfs.reduced_domain(), 1000)[:-1]

    (
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
    ) = util.signal_param_guess(5)

    sig = pdfs.normalised_signal(
        centres,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        pdfs.domain(),
    )
    reduced_sig = pdfs.normalised_signal(
        reduced_centres,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        pdfs.reduced_domain(),
    )

    bkg, reduced_bkg, bkg_params = _bkg(
        centres, reduced_centres, alt_bkg, year, sign, magnetisation, bdt_cut
    )

    # TODO tmp hack while _bkg doesnt return actual alt bkg params
    if alt_bkg:
        bkg_params = []

    pdf = _pdf(
        centres,
        alt_bkg,
        (centre, width_l, width_r, alpha_l, alpha_r, beta, *bkg_params),
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(centres, sig, color="b", label="full")
    ax[0].plot(centres, bkg, color="b")

    ax[0].plot(reduced_centres, reduced_sig, linestyle="--", color="r", label="reduced")
    ax[0].plot(reduced_centres, reduced_bkg, linestyle="--", color="r")
    ax[0].legend()

    ax[1].plot(centres, pdf)

    title = "Alternative Bkg" if alt_bkg else "Signal + Bkg"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alt_bkg",
        action="store_true",
        help="Plot the alt bkg model with the D-pi combination",
    )
    parser.add_argument(
        "--year",
        type=str,
        help="data taking year for alt bkg dump",
    )
    parser.add_argument(
        "--sign",
        type=str,
        choices={"cf", "dcs"},
        help="sign for alt bkg dump",
    )
    parser.add_argument(
        "--magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction for alt bkg dump",
    )
    parser.add_argument(
        "--bdt_cut",
        action="store_true",
        help="Whether to use the bdt cut for alt bkg dump",
    )

    main(**vars(parser.parse_args()))
