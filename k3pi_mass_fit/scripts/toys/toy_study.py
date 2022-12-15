"""
Toy study for mass fit

"""
import sys
import pathlib
from typing import Callable, Tuple
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils
from lib_data import stats


def _pull(
    rng: np.random.Generator, n_sig: int, n_bkg: int, sign: str, time_bin: int
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    combined, true_params = toy_utils.gen_points(rng, n_sig, n_bkg, sign, time_bin)

    # Perform fit
    sig_frac = true_params[0]
    n_sig = len(combined) * sig_frac
    n_bkg = len(combined) * true_params[1]

    true_params = np.array((n_sig, n_bkg, *true_params[2:]))

    bins = np.linspace(*pdfs.domain(), 100)
    counts, _ = stats.counts(combined, bins)
    fitter = fit.binned_fit(counts, bins, sign, time_bin, sig_frac)

    fit_params = fitter.values
    fit_errs = fitter.errors

    pull = (true_params - fit_params) / fit_errs

    # Keeping this in in case I want to plot later for debug
    if False and (np.abs(pull) > 10).any():

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.model(x, *fit_params)

        pts = np.linspace(*pdfs.domain(), 250)
        centres = (pts[1:] + pts[:-1]) / 2

        fig, ax = plt.subplots()
        ax.plot(centres, stats.areas(pts, fitted_pdf(pts)), "r--")
        ax.hist(combined, bins=pts)
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
    rng = np.random.default_rng()

    n_params = 10
    return_vals = tuple([] for _ in range(n_params))
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
        np.ndarray,
    ],
    path: str,
) -> None:
    """
    Plot pulls

    """
    fig, ax = plt.subplots(4, 3, figsize=(12, 9))
    labels = (
        "n signal",
        "n bkg",
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
    n_sig, n_bkg = 320000, 160000

    out_list = Manager().list()

    n_procs = 8
    n_experiments = 50
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


def main():
    """
    do a pull study

    """
    _do_pull_study()


if __name__ == "__main__":
    main()
