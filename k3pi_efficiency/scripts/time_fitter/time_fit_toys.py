"""
Generate toys from the time fitter PDF

"""
import sys
import pathlib
from typing import Callable, Tuple
from multiprocessing import Manager, Process
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_efficiency import time_fitter


def _noise(gen: np.random.Generator, *args: float) -> Tuple[float, ...]:
    """
    Add some random noise to some floats

    """
    noise = 0.0
    return tuple(
        noise * val
        for noise, val in zip(
            [(1 - noise) + 2 * noise * gen.random() for _ in args], args
        )
    )


def _gen(
    rng: np.random.Generator,
    n_gen: int,
    plot=False,
) -> np.ndarray:
    """
    Generate samples from a pdf

    """
    if not n_gen:
        return np.array([])

    args = np.array(_noise(rng, 0.5, 1.0, 2.5, 1.0, 2.5, 1))

    scale = 1
    x = rng.exponential(scale=scale, size=n_gen)
    max_x = np.max(x)

    def gen_func(x):
        """ Exponential to draw from """
        return 0.25 * np.exp(-scale * x)

    y = gen_func(x) * rng.random(n_gen)

    f_eval = time_fitter.pdf(x, *args)

    # Evaluate the function at a lot of points
    pts = np.linspace(0, max_x, 1000)

    # Explicitly check that all the generating PDF is strictly
    # greater than the function
    assert (time_fitter.pdf(pts, *args) < gen_func(pts)).all()

    keep = y < f_eval

    if plot:
        _, ax = plt.subplots()

        ax.plot(pts, time_fitter.pdf(pts, *args))
        ax.plot(pts, gen_func(pts), alpha=0.5, color="r")
        ax.scatter(x[keep], y[keep], c="k", marker=".")
        ax.scatter(x[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return x[keep], args


def _pull(rng: np.random.Generator, n_gen: int) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    times, true_params = _gen(rng, n_gen, plot=False)

    # Perform fit
    fitter = time_fitter.fit(times, true_params)

    fit_params = fitter.values
    fit_errs = fitter.errors

    pull = (true_params - fit_params) / fit_errs

    # Keeping this in in case I want to plot later for debug
    if (np.abs(pull) > 10).any() or True:

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return time_fitter.normalised_pdf(x, *fitter.values)[1]

        pts = np.linspace(0, 10, 250)
        fig, ax = plt.subplots()
        ax.plot(pts, fitted_pdf(pts), "r--")
        ax.hist(times, bins=250, density=True)
        fig.suptitle("toy data")
        fig.savefig("toy.png")
        plt.show()

    return pull


def _pull_study(
    n_experiments: int,
    n_gen: int,
    out_list: list,
) -> None:
    """
    Return arrays of pulls for the 6 fit parameters

    return value appended to out_list; (6xN) shape array of pulls

    """
    rng = np.random.default_rng()

    return_vals: tuple = tuple([] for _ in range(6))

    for _ in tqdm(range(n_experiments)):
        for lst, val in zip(return_vals, _pull(rng, n_gen)):
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
    ],
    path: str,
) -> None:
    """
    Plot pulls

    """
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    labels = ("t0", "n", "m", "a", "b", "k")

    for a, p, l in zip(ax.ravel(), pulls, labels):
        a.hist(p, label=f"{np.mean(p):.4f}+-{np.std(p):.4f}", bins=np.linspace(-10, 10, 21))
        a.set_title(l)
        a.legend()

    fig.savefig(path)
    fig.tight_layout()
    plt.show()


def main():
    """
    Generate some toy times from our PDF, fit them, show pulls

    """
    n_gen = 250000

    out_list = Manager().list()

    n_procs = 8
    n_experiments = 15
    procs = [
        Process(
            target=_pull_study,
            args=(n_experiments, n_gen, out_list),
        )
        for _ in range(n_procs)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    pulls = np.concatenate(out_list, axis=1)

    _plot_pulls(pulls, f"pulls_{n_gen=}_{n_procs=}_{n_experiments=}.png")


if __name__ == "__main__":
    main()
