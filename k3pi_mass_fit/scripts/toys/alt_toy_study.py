"""
Pull studies for simultaneous mass fitter with
alternate background model

"""
import os
import sys
import time
import pathlib
import argparse
from typing import Tuple, Callable
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, util, plotting, definitions, bkg
from lib_data import stats


class InvalidFitError(Exception):
    """
    Invalid fit
    """


def _bins(n_underflow: int = 3) -> np.ndarray:
    """
    For fitting + plotting

    :param n_underflow: number of underflow bins
    :returns: bins

    """
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()

    return definitions.nonuniform_mass_bins(
        (gen_low, fit_low, gen_high), (n_underflow, 150)
    )


def _plot_fit(
    fit_params,
    rs_combined,
    ws_combined,
    rs_fit_axes,
    ws_fit_axes,
    rs_bkg_pdf,
    ws_bkg_pdf,
):
    """
    Plot fit on an axis

    """
    bins = _bins()

    for axes, data, params, pdf in zip(
        (rs_fit_axes, ws_fit_axes),
        (rs_combined, ws_combined),
        util.alt_rs_ws_params(fit_params),
        (rs_bkg_pdf, ws_bkg_pdf),
    ):
        plotting.alt_bkg_fit(
            axes,
            *stats.counts(data, bins),
            bins,
            pdf,
            pdfs.reduced_domain(),
            params,
        )


def _sig_params():
    """
    Parameters for signal peak (RS)

    """
    return util.signal_param_guess(time_bin=5)


def _bkg_params(sign: str):
    """
    Parameters for alt bkg

    """
    assert sign in {"dcs", "cf"}
    if sign == "dcs":
        return (0, 0.001, 0)

    return (0, 0, 0)


def _gen(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    bkg_pdf: Callable,
    sign: str,
):
    """
    Generate points

    Same signal model, slightly different bkg models

    """
    sig = toy_utils.gen_sig(rng, n_sig, _sig_params(), pdfs.reduced_domain())

    # Get bkg pdf across the whole generator region
    bkg_pts = toy_utils.gen_alt_bkg(
        rng, n_bkg, bkg_pdf, _bkg_params(sign), pdfs.reduced_domain()
    )

    return np.concatenate((sig, bkg_pts))


def _pull(
    rng: np.random.Generator,
    bins: np.ndarray,
    n_underflow: int,
    rs_bkg_pdf: Callable,
    ws_bkg_pdf: Callable,
    out_dict: dict,
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    rs_n_sig, ws_n_sig, n_bkg = 800_000, 2_200, 25_000

    # Generate stuff
    rs_combined = _gen(rng, rs_n_sig, n_bkg, rs_bkg_pdf, "cf")
    ws_combined = _gen(rng, ws_n_sig, n_bkg, ws_bkg_pdf, "dcs")

    # Perform fit
    true_params = np.array(
        (
            rs_n_sig,
            n_bkg,
            ws_n_sig,
            n_bkg,
            *_sig_params(),
            *_bkg_params("cf"),
            *_bkg_params("dcs"),
        )
    )
    rs_counts, _ = stats.counts(rs_combined, bins)
    ws_counts, _ = stats.counts(ws_combined, bins)

    fitter = fit.alt_simultaneous_fit(
        rs_counts[n_underflow:],
        ws_counts[n_underflow:],
        bins[n_underflow:],
        true_params,
        rs_bkg_pdf,
        ws_bkg_pdf,
    )
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

    return pull


def _pull_study(
    bins: np.ndarray,
    n_underflow: int,
    n_experiments: int,
    rs_bkg_pdf: Callable,
    ws_bkg_pdf: Callable,
    out_list: list,
    out_dict: dict,
) -> None:
    """
    Return arrays of pulls for the 9 fit parameters

    return value appended to out_list; (9xN) shape array of pulls

    """
    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)
    n_params = 16

    return_vals = tuple([] for _ in range(n_params))
    for _ in tqdm(range(n_experiments)):
        while True:
            try:
                pulls = _pull(
                    rng,
                    bins,
                    n_underflow,
                    rs_bkg_pdf,
                    ws_bkg_pdf,
                    out_dict,
                )
                break
            except InvalidFitError:
                print("invalid fit")

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

    # Need to do this for the violin plot
    pulls = list(pulls)

    # Find quartiles
    medians = np.median(pulls, axis=1)

    stds = np.std(pulls, axis=1)
    means = np.mean(pulls, axis=1)
    low_sig = means - stds
    high_sig = means + stds

    pull_axis.scatter(
        medians,
        positions,
        marker="o",
        s=30,
        zorder=3,
        label="Median",
        edgecolors="k",
        facecolors="w",
    )
    pull_axis.hlines(
        positions,
        low_sig,
        high_sig,
        color="k",
        linestyle="-",
        lw=3,
        label=r"$\mu \pm \sigma$",
    )

    parts = pull_axis.violinplot(
        pulls,
        positions,
        vert=False,
        showmeans=True,
        points=500,
        showextrema=False,
    )
    for part in parts["bodies"]:
        part.set_edgecolor("black")

    pull_axis.set_yticks(positions)

    pull_axis.legend()

    ylabels = [
        f"{label}\n{mean:.3f}" + r"$\pm$" + f"{std:.3f}"
        for (label, mean, std) in zip(labels, means, stds)
    ]

    pull_axis.set_yticklabels(ylabels)

    pull_axis.axvline(0.0, color="k")
    for val in (-1.0, 1.0):
        pull_axis.axvline(val, color="k", alpha=0.5, linestyle="--")

    fig.suptitle(f"{n_experiments=}")


def _do_pull_study():
    """
    Do the pull study

    """
    manager = Manager()
    out_list = manager.list()
    out_dict = manager.dict()

    n_underflow = 3
    bins = _bins(n_underflow)

    rs_bkg_pdf = bkg.pdf(bins[n_underflow:], "2018", "magdown", "cf", bdt_cut=True)
    ws_bkg_pdf = bkg.pdf(bins[n_underflow:], "2018", "magdown", "dcs", bdt_cut=True)

    # For plotting pulls
    fig, axes = plt.subplot_mosaic(
        "AAAA\nAAAA\nAAAA\nAAAA\nAAAA\nCCDD\nCCDD\nEEFF\nEEFF", figsize=(8, 10)
    )

    n_procs = 5
    n_experiments = 5
    procs = [
        Process(
            target=_pull_study,
            args=(
                bins,
                n_underflow,
                n_experiments,
                rs_bkg_pdf,
                ws_bkg_pdf,
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
        (axes["C"], axes["E"]),
        (axes["D"], axes["F"]),
        rs_bkg_pdf,
        ws_bkg_pdf,
    )
    for axis in (axes["C"], axes["D"]):
        axis.set_title("Example fit")
    fig.tight_layout()

    n_bins = len(_bins()) - 1
    path = f"alt_simul_pulls_{n_procs=}_{n_experiments=}_{n_bins=}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    plt.show()
    plt.close(fig)


def main():
    """
    do a pull study

    """
    parser = argparse.ArgumentParser(
        description="Toy pull study for mass fitter using the alt bkg model"
    )

    _do_pull_study(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
