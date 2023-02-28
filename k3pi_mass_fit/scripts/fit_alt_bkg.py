"""
Fit the alt bkg model with the sqrt bkg model

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from libFit import toy_utils, pdfs, util, bkg


def _fit_fcn(x, a, b):
    """Used for the fitter + plotting"""
    return pdfs.normalised_bkg(x, a, b, pdfs.domain())


def main(*, sign: str):
    """
    Generate points according to the alt bkg model
    Perform a fit to them using the sqrt model

    """
    # Find the bkg pdf
    pdf_domain = pdfs.domain()
    bins = np.linspace(*pdfs.domain(), 200)
    bkg_pdf = bkg.pdf(bins, "2018", "magdown", sign, bdt_cut=False)

    # Generate points
    rng = np.random.default_rng()
    points = toy_utils.gen_alt_bkg(
        rng, 500_000, bkg_pdf, (0, 0, 0), pdf_domain, verbose=True
    )

    fig, axis = plt.subplots()
    axis.hist(points, bins)

    # Perform a fit to them
    cost = UnbinnedNLL(points, _fit_fcn)
    a_guess, b_guess = util.sqrt_bkg_param_guess(sign)

    fitter = Minuit(cost, a=a_guess, b=b_guess)
    fitter.migrad()

    fit_a, fit_b = np.array(fitter.values)
    err_a, err_b = np.array(fitter.errors)

    axis.plot(
        bins, (bins[1] - bins[0]) * len(points) * _fit_fcn(bins, fit_a, fit_b), "k-"
    )

    axis.set_xlabel(r"$\Delta M$")
    fig.suptitle(f"a={fit_a:.5f}$\\pm${err_a:.5f}\t{fit_b:.5f}$\\pm${err_b:.5f}")
    fig.tight_layout()

    path = f"fit_alt_bkg_{sign}.png"
    print(f"saving {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the phenomenological background model with the sqrt model"
    )
    parser.add_argument("sign", choices={"cf", "dcs"}, help="Which bkg model to use")

    main(**vars(parser.parse_args()))
