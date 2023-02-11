"""
Pull studies for simultaneous mass fitter

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

from libFit import pdfs, fit, toy_utils, definitions, util
from lib_data import stats


class InvalidFitError(Exception):
    """
    Invalid fit
    """


def _bins():
    """
    For fitting + plotting

    """
    return definitions.mass_bins(200)


def _plot_fit(fit_params, rs_combined, ws_combined, fit_axes):
    """
    Plot fit on an axis

    """
    for axis, data, params in zip(
        fit_axes, (rs_combined, ws_combined), util.rs_ws_params(fit_params)
    ):

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.model(x, *params)

        pts = _bins()
        centres = (pts[1:] + pts[:-1]) / 2
        widths = pts[1:] - pts[:-1]

        axis.plot(centres, widths * fitted_pdf(centres), "r--")
        axis.hist(data, bins=pts)


def _pull(
    rng: np.random.Generator,
    rs_n_sig: int,
    ws_n_sig: int,
    n_bkg: int,
    time_bin: int,
    out_dict: dict,
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    # Perform fit
    rs_combined, rs_true_params = toy_utils.gen_points(
        rng,
        rs_n_sig,
        n_bkg,
        "cf",
        time_bin,
        pdfs.background_defaults("cf"),
        verbose=False,
    )
    ws_combined, ws_true_params = toy_utils.gen_points(
        rng,
        ws_n_sig,
        n_bkg,
        "dcs",
        time_bin,
        pdfs.background_defaults("dcs"),
        verbose=False,
    )

    true_params = np.array(util.fit_params(rs_true_params, ws_true_params))

    bins = _bins()
    rs_counts, _ = stats.counts(rs_combined, bins)
    ws_counts, _ = stats.counts(ws_combined, bins)

    fitter = fit.binned_simultaneous_fit(rs_counts, ws_counts, bins, time_bin)
    if not fitter.valid:
        raise InvalidFitError

    fit_params = fitter.values
    fit_errs = fitter.errors

    if not out_dict:
        # If the output dict is empty, this is the first pseudoexperiment
        # Store the counts and fit params in the dict in this case
        # so that we can plot it later
        out_dict["fit_params"] = np.array(fit_params)
        out_dict["rs_combined"] = rs_combined
        out_dict["ws_combined"] = ws_combined

        # For plotting
        out_dict["labels"] = fitter.parameters

    pull = (true_params - fit_params) / fit_errs

    if abs(pull[0]) > 20:
        print(fitter)
        # Just raise
        raise InvalidFitError(f"{pull[0]=}")

    return pull


def _pull_study(
    n_experiments: int,
    n_evts: Tuple[int, int],
    time_bin: int,
    out_list: list,
    out_dict: dict,
) -> None:
    """
    Return arrays of pulls for the 9 fit parameters

    return value appended to out_list; (9xN) shape array of pulls

    """
    n_rs_sig, n_ws_sig, n_bkg = n_evts

    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    # 10 params for the simultaneous sqrt bkg fit
    n_params = 14

    return_vals = tuple([] for _ in range(n_params))
    for _ in tqdm(range(n_experiments)):
        while True:
            try:
                pulls = _pull(
                    rng,
                    n_rs_sig,
                    n_ws_sig,
                    n_bkg,
                    time_bin,
                    out_dict,
                )
                break
            except InvalidFitError:
                print("invalid fit")

        # Only look at the pulls of n_sig and n_bkg
        for pull, lst in zip(pulls, return_vals):
            lst.append(pull)

    out_list.append(np.array(return_vals))


def _plot_pulls(
    fig,
    pull_axis,
    pulls: Tuple,
    n_experiments: int,
    labels: Tuple,
) -> None:
    """
    Plot pulls

    """
    positions = tuple(range(1, len(labels) + 1))
    pull_axis.violinplot(
        list(pulls),
        positions,
        vert=False,
        showmeans=True,
        points=500,
        showextrema=True,
    )
    pull_axis.set_yticks(positions)
    means = np.mean(pulls, axis=1)
    stds = np.std(pulls, axis=1)

    ylabels = [
        f"{label}\n{mean:.3f}" + r"$\pm$" + f"{std:.3f}"
        for (label, mean, std) in zip(labels, means, stds)
    ]

    pull_axis.set_yticklabels(ylabels)

    pull_axis.axvline(0.0, color="k")

    fig.suptitle(f"{n_experiments=}")


def _do_pull_study():
    """
    Do the pull study

    """
    time_bin = 5
    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_rs_sig, n_ws_sig, n_bkg = 18_000_000, 55000, 50000

    manager = Manager()
    out_list = manager.list()
    out_dict = manager.dict()

    # For plotting pulls
    fig, axes = plt.subplot_mosaic("AAAA\nAAAA\nAAAA\nCCDD\nCCDD", figsize=(8, 10))

    n_procs = 6
    n_experiments = 10
    procs = [
        Process(
            target=_pull_study,
            args=(
                n_experiments,
                (n_rs_sig, n_ws_sig, n_bkg),
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
        out_dict["labels"],
    )

    _plot_fit(
        out_dict["fit_params"],
        out_dict["rs_combined"],
        out_dict["ws_combined"],
        (axes["C"], axes["D"]),
    )
    for axis in (axes["C"], axes["D"]):
        axis.set_title("Example fit")
    fig.tight_layout()

    path = f"simul_pulls_{n_procs=}_{n_experiments=}.png"
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
