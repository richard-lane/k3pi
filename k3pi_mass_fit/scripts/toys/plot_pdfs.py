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

from libFit import pdfs, definitions


def _bkg(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Background evaluated at some points

    """
    if not args.alt_bkg:
        bkg_params = pdfs.background_defaults("RS")
        return pdfs.normalised_bkg(points, *bkg_params), bkg_params

    low, high = pdfs.domain()

    def bkg_pdf(x):
        """toy bkg pdf"""
        return 3 * np.sqrt(x - low) / (2 * np.sqrt(high - low))

    bkg_params = bkg_pdf, 0.0, 0.1, 0.001
    return pdfs.estimated_bkg(points, *bkg_params), bkg_params


def _pdf(points: np.ndarray, args: argparse.Namespace, params: Tuple) -> np.ndarray:
    """
    Combined sig + bkg pdf evaluated at some points

    """
    n_sig = 0.2
    n_bkg = 1 - n_sig

    if not args.alt_bkg:
        return pdfs.model(points, n_sig, n_bkg, *params)

    return pdfs.model_alt_bkg(points, n_sig, n_bkg, *params)


def main(args):
    """Make and show a plot"""
    centres = definitions.mass_bins(1000)

    (
        centre,
        width_l,
        alpha_l,
        beta,
    ) = pdfs.signal_defaults(5)
    width_r = 1.5 * width_l
    alpha_r = 1.5 * alpha_l

    sig = pdfs.normalised_signal(
        centres,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
    )

    bkg, bkg_params = _bkg(centres, args)
    pdf = _pdf(
        centres,
        args,
        (centre, width_l, width_r, alpha_l, alpha_r, beta, *bkg_params),
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(centres, sig)
    ax[0].plot(centres, bkg)

    ax[1].plot(centres, pdf)

    title = "Alternative Bkg" if args.alt_bkg else "Signal + Bkg"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt_bkg", action="store_true")

    main(parser.parse_args())
