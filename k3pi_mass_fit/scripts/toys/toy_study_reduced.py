"""
Toy study for mass fit

"""
import os
import sys
import time
import pathlib
import argparse
from typing import Tuple
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, definitions
from lib_data import stats


class InvalidFitError(Exception):
    """
    Invalid fit
    """


def _plot_fit(fit_params, combined, fit_axis):
    """
    Plot fit on an axis

    """

    def fitted_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.model_reduced(x, *fit_params)

    pts = definitions.reduced_mass_bins(100)
    centres = (pts[1:] + pts[:-1]) / 2
    widths = pts[1:] - pts[:-1]

    fit_axis.plot(centres, widths * fitted_pdf(centres), "r--")
    fit_axis.hist(combined, bins=pts)


def _pull(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    time_bin: int,
    out_dict: dict,
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    # Perform fit
    combined, true_params = toy_utils.gen_points_reduced(
        rng,
        n_sig,
        n_bkg,
        sign,
        time_bin,
        pdfs.background_defaults(sign),
        verbose=False,
    )

    true_params = np.array(true_params)
    sig_frac = true_params[0] / (true_params[0] + true_params[1])

    bins = definitions.reduced_mass_bins(100)
    counts, _ = stats.counts(combined, bins)

    fitter = fit.binned_fit_reduced(counts, bins, sign, time_bin, sig_frac)
    if not fitter.valid:
        raise InvalidFitError

    fit_params = fitter.values
    fit_errs = fitter.errors

    if not out_dict:
        out_dict["fit_params"] = np.array(fit_params)
        out_dict["combined"] = combined

    pull = (true_params - fit_params) / fit_errs

    if abs(pull[0]) > 20:
        print(fitter)
        # Just raise
        raise InvalidFitError(f"{pull[0]=}")

    return pull


def _pull_study(
    n_experiments: int,
    n_evts: Tuple[int, int],
    sign: str,
    time_bin: int,
    out_list: list,
    out_dict: dict,
) -> None:
    """
    Return arrays of pulls for the 9 fit parameters

    return value appended to out_list; (9xN) shape array of pulls

    """
    n_sig, n_bkg = n_evts

    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    # 10 params for the sqrt bkg fit
    n_params = 10

    return_vals = tuple([] for _ in range(n_params))
    for _ in tqdm(range(n_experiments)):
        while True:
            try:
                pulls = _pull(
                    rng,
                    n_sig,
                    n_bkg,
                    sign,
                    time_bin,
                    out_dict,
                )
                break
            except InvalidFitError:
                print("invalid fit")

        # Only look at the pulls of n_sig and n_bkg
        return_vals[0].append(pulls[0])
        return_vals[1].append(pulls[1])
        for lst in return_vals[2:]:
            lst.append(np.nan)

    out_list.append(np.array(return_vals))


def _plot_pulls(
    fig,
    axis,
    pulls: Tuple,
    n_experiments: int,
) -> None:
    """
    Plot pulls

    """
    labels = (
        "n signal",
        "n bkg",
    )

    positions = (1, 2)
    axis.violinplot(
        list(pulls[0:2]),
        positions,
        vert=False,
        showmeans=True,
        points=500,
        showextrema=True,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(("N sig", "N bkg"))

    axis.axvline(0.0, color="k")

    # pull = pulls[i]

    # axis.hist(
    #     pull,
    #     label=f"{np.mean(pull):.4f}+-{np.std(pull):.4f}",
    #     bins=100,  # np.linspace(-6, 6, 120),
    # )
    # axis.set_title(labels[i])
    # axis.legend()

    fig.suptitle(f"reduced bkg fit, {n_experiments=}")


def _do_pull_study():
    """
    Do the pull study

    """
    sign, time_bin = "cf", 5
    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 50000, 50000

    manager = Manager()
    out_list = manager.list()
    out_dict = manager.dict()

    # For plotting pulls
    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB\nBBB", figsize=(5, 8))

    n_procs = 6
    n_experiments = 200
    procs = [
        Process(
            target=_pull_study,
            args=(
                n_experiments,
                (n_sig, n_bkg),
                sign,
                time_bin,
                out_list,
                out_dict,
            ),
        )
        for _ in range(n_procs)
    ]

    for p in procs:
        p.start()

        # So the processes start at a different time and have different seeds
        time.sleep(1)

    for p in procs:
        p.join()

    pulls = np.concatenate(out_list, axis=1)

    _plot_pulls(
        fig,
        axes["A"],
        pulls,
        n_procs * n_experiments,
    )

    _plot_fit(
        out_dict["fit_params"],
        out_dict["combined"],
        axes["B"],
    )
    axes["B"].set_title("Example fit")
    fig.tight_layout()

    path = f"reduced_pulls_{sign}_{time_bin=}_{n_sig=}_{n_bkg=}_{n_procs=}_{n_experiments=}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    plt.show()
    plt.close(fig)


def main():
    """
    do a pull study

    """
    parser = argparse.ArgumentParser(
        description="Toy pull study for mass fitter using the reduced background"
    )

    _do_pull_study(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
