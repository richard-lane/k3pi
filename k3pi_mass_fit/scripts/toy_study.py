"""
Toy study for mass fit

"""
import sys
import pathlib
import argparse
from typing import Callable, Tuple
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def _pull(
    rng: np.random.Generator, n_sig: int, n_bkg: int, sign: str, time_bin: int
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    combined, true_params = _gen_points(rng, n_sig, n_bkg, sign, time_bin)

    # Perform fit
    sig_frac = true_params[0]
    fitter = fit.fit(combined, sign, time_bin, sig_frac)

    fit_params = fitter.values
    fit_errs = fitter.errors

    pull = (true_params - fit_params) / fit_errs

    # Keeping this in in case I want to plot later for debug
    if (np.abs(pull) > 10).any():

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.fractional_pdf(x, *fit_params)

        pts = np.linspace(*pdfs.domain(), 250)
        fig, ax = plt.subplots()
        ax.plot(pts, fitted_pdf(pts), "r--")
        ax.hist(combined, bins=250, density=True)
        fig.suptitle("toy data")
        fig.savefig("toy.png")
        plt.show()
    return pull


def _pull_study(
    n_experiments: int,
    n_evts: Tuple[int, int],
    sign: str,
    time_bin: int,
    out_list: list,
) -> None:
    """
    Return arrays of pulls for the 9 fit parameters

    return value appended to out_list; (9xN) shape array of pulls

    """
    n_sig, n_bkg = n_evts

    # TODO maybe need to be more careful about seeding this, since we're multiprocessing...
    rng = (
        np.random.default_rng()
    )  # seed=3 makes the fitter set some of the parameters to NaN

    return_vals: tuple = tuple([] for _ in range(9))
    for _ in tqdm(range(n_experiments)):
        for lst, val in zip(return_vals, _pull(rng, n_sig, n_bkg, sign, time_bin)):
            lst.append(val)

    out_list.append(np.array(return_vals))


def _plot_pulls(
    pulls: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    path: str,
) -> None:
    """
    Plot pulls

    """
    fig, ax = plt.subplots(3, 3, figsize=(12, 8))
    labels = (
        "signal fraction",
        "centre",
        "width L",
        "width R",
        r"$\alpha_L$",
        r"$\alpha_R$",
        r"$\beta$",
        "a",
        "b",
    )

    for a, p, l in zip(ax.ravel(), pulls, labels):
        a.hist(p, label=f"{np.mean(p):.4f}+-{np.std(p):.4f}", bins=20)
        a.set_title(l)
        a.legend()

    fig.savefig(path)
    fig.tight_layout()
    plt.show()


def _do_pull_study():
    """
    Do the pull study

    """
    sign, time_bin = "RS", 5

    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 160000, 80000

    out_list = Manager().list()

    n_procs = 6
    n_experiments = 15
    procs = [
        Process(
            target=_pull_study,
            args=(n_experiments, (n_sig, n_bkg), sign, time_bin, out_list),
        )
        for _ in range(n_procs)
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    pulls = np.concatenate(out_list, axis=1)

    _plot_pulls(
        pulls,
        f"pulls_{sign}_{time_bin=}_{n_sig=}_{n_bkg=}_{n_procs=}_{n_experiments=}.png",
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
    fitter = fit.fit(combined, sign, time_bin, sig_frac)

    def fitted_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.fractional_pdf(x, *fitter.values)

    pts = np.linspace(*pdfs.domain(), 250)
    fig, ax = plt.subplots()
    ax.plot(pts, fitted_pdf(pts), "r--")
    ax.hist(combined, bins=250, density=True)
    fig.suptitle("toy data")
    plt.savefig("toy_mass_fit.png")


def main(args: argparse.Namespace):
    """
    Either do a pull study, or just do 1 fit

    """
    if args.pull:
        _do_pull_study()

    else:
        _toy_fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Whether to do a full pull study, or just plot one fit",
    )

    main(parser.parse_args())
