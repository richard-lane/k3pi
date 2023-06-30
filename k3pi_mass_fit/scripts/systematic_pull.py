"""
Pull study to assess systematic associated with the mass fitter

"""
import os
import sys
import time
import pickle
import pathlib
from multiprocessing import Process, Manager
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from lib_data import stats
from lib_data import get, training_vars
from lib_cuts.get import classifier as get_clf
from lib_cuts.definitions import THRESHOLD
from libFit import util as mass_util, fit, plotting, pdfs, definitions, toy_utils


class GaussKDE:
    """
    Callable for KDE

    """

    def __init__(self, positions: np.ndarray, counts: np.ndarray, width: float):
        """
        Tell us where to put the Gaussians and how strong to make them

        """
        assert len(positions) == len(counts)

        self._gaussians = [norm(loc=position, scale=width) for position in positions]
        self._counts = counts
        self._total = np.sum(counts)

    def __call__(self, points: np.ndarray):
        """
        Evaluate the estimated PDF

        """
        retval = np.zeros_like(points)

        # Could vectorise
        for gaussian, count in zip(self._gaussians, self._counts):
            retval += count * gaussian.pdf(points)

        # Scale so the KDE is normalised
        retval /= self._total

        return retval


def _sig_counts(
    year: str, sign: str, magnetisation: str, bins: np.ndarray
) -> np.ndarray:
    """
    binned MC counts

    :returns: array of signal delta masses

    """
    # Read the right dataframe
    mc_df = get.particle_gun(year, sign, magnetisation, show_progress=True)

    # Perform BDT cut
    clf = get_clf(year, "dcs", magnetisation)

    training_labels = list(training_vars.training_var_names())
    sig_predictions = clf.predict_proba(mc_df[training_labels])[:, 1] > THRESHOLD

    mc_df = mc_df[sig_predictions]
    print(len(mc_df))

    count, _ = stats.counts(mass_util.delta_m(mc_df), bins)
    print(f"total sig: {np.sum(count)}")

    # Removed this because it just changes the peak shape
    # Smooth out with a KDE
    # Assume bins are constant width past the underflow
    # centres = (bins[:-1] + bins[1:]) / 2
    # kde = GaussKDE(centres[n_underflow:], count[n_underflow:], 0.6 * (bins[-1] - bins[-2]))

    # Don't use the KDE because I don't like it
    return count


def _bkg(
    rng: np.random.Generator, n_gen: int, bins: np.ndarray, sign: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Background counts with Poisson errors from the PDF

    """
    points = toy_utils.gen_bkg_sqrt(
        rng, n_gen, mass_util.sqrt_bkg_param_guess(sign), pdfs.reduced_domain()
    )

    counts, _ = stats.counts(points, bins)

    return counts


def _sig(
    rng: np.random.Generator, sig_counts: np.ndarray, n_tot: int, n_underflow: int
) -> np.ndarray:
    """
    Array of signal counts after Poisson fluctuations

    """
    # Scale such that we have the right counts
    # We want the number in the fit region to equal the number requested
    scale_factor = n_tot / np.sum(sig_counts[n_underflow:])
    print(f"{scale_factor=}", end="\t")
    counts = sig_counts * scale_factor

    # Apply Poisson fluctuations
    return rng.poisson(lam=counts).astype(np.float64)


def _pull_study(
    n_experiments: int,
    out_list: list,
    out_dict: dict,
    rs_signal_counts: np.ndarray,
    ws_signal_counts: np.ndarray,
    bins: np.ndarray,
    n_underflow: int,
    fit_range: Tuple[float, float],
):
    """
    Generate bkg using accept-reject

    Find signal counts from MC and apply Poisson fluctuations

    """
    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    # we want the signal counts to add up to these
    n_rs_sig = 400_000
    n_ws_sig = 6_000

    # number to generate for the bkg
    n_rs_bkg = 100_000
    n_ws_bkg = 80_000

    # want to track n_sig and n_bkg for both RS and WS
    pulls = [np.full(n_experiments, np.inf, dtype=float) for _ in range(4)]
    for i in tqdm(range(n_experiments)):
        # Keep trying until a fit converges
        while True:
            # Generate bkg
            rs_bkg = _bkg(rng, n_rs_bkg, bins=bins, sign="cf")
            ws_bkg = _bkg(rng, n_ws_bkg, bins=bins, sign="dcs")

            # Add fluctuations to sig
            rs_sig = _sig(rng, rs_signal_counts, n_rs_sig, n_underflow)
            ws_sig = _sig(rng, ws_signal_counts, n_ws_sig, n_underflow)

            # Combine
            rs_counts = rs_sig + rs_bkg
            ws_counts = ws_sig + ws_bkg

            # Do a fit
            initial_guess = (
                np.sum(rs_sig),
                np.sum(rs_bkg),
                np.sum(ws_sig),
                np.sum(ws_bkg),
                *mass_util.signal_param_guess(),
                *mass_util.sqrt_bkg_param_guess("cf"),
                *mass_util.sqrt_bkg_param_guess("dcs"),
            )
            binned_fitter = fit.binned_simultaneous_fit(
                rs_counts[n_underflow:],
                ws_counts[n_underflow:],
                bins[n_underflow:],
                initial_guess,
                fit_range,
            )
            if not binned_fitter.valid:
                print("fit not valid")
                continue

            # Find the pulls
            fit_vals = binned_fitter.values
            fit_errs = binned_fitter.errors
            rs_sig_pull = (fit_vals[0] - n_rs_sig) / fit_errs[0]
            rs_bkg_pull = (fit_vals[1] - n_rs_bkg) / fit_errs[1]
            ws_sig_pull = (fit_vals[2] - n_ws_sig) / fit_errs[2]
            ws_bkg_pull = (fit_vals[3] - n_ws_bkg) / fit_errs[3]

            pulls[0][i] = rs_sig_pull
            pulls[1][i] = rs_bkg_pull
            pulls[2][i] = ws_sig_pull
            pulls[3][i] = ws_bkg_pull

            # For plotting
            if not out_dict:
                print("storing fit params")
                out_dict["fit_params"] = np.array(fit_vals)
                out_dict["rs_count"] = rs_counts
                out_dict["ws_count"] = ws_counts
                out_dict["labels"] = binned_fitter.parameters[:4]

            break

    out_list.append(np.array(pulls))


def _plot_fit(
    fit_params, rs_count, ws_count, rs_fit_axes, ws_fit_axes, bins, fit_range
):
    """
    Plot fit on an axis

    """
    for axes, counts, params in zip(
        (rs_fit_axes, ws_fit_axes),
        (rs_count, ws_count),
        mass_util.rs_ws_params(fit_params),
    ):
        plotting.mass_fit(axes, counts, np.sqrt(counts), bins, fit_range, params)


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
    # Only plot n sig pulls
    pulls = pulls[0], pulls[2]
    labels = labels[0], labels[2]

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
        list(pulls),
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
    pull_axis.set_xlim(-5.0, 5.0)
    pull_axis.set_xlabel("Pull")

    fig.suptitle(f"{n_experiments=}")


def main():
    """
    Get the signal models from MC and the background model from a pickle dump

    Use KDE to smooth the background out a bit, then combine them to get
    ideal WS + RS distributions

    Then apply Poisson fluctuations and scale so the total counts are approximately
    what we'd expect in the data

    Finally, fit and see if we can extract the right numbers of signal and background events

    """
    # Get MC counts from the dataframes
    year, magnetisation = "2018", "magdown"
    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], high), (n_underflow, 150)
    )

    # Get the MC counts in all the bins (including underflow)
    rs_count = _sig_counts(year, "cf", magnetisation, bins)
    ws_count = _sig_counts(year, "dcs", magnetisation, bins)

    manager = Manager()
    out_dict = manager.dict()
    out_list = manager.list()
    n_procs = 4
    n_experiments = 25
    procs = [
        Process(
            target=_pull_study,
            args=(
                n_experiments,
                out_list,
                out_dict,
                rs_count,
                ws_count,
                bins,
                n_underflow,
                fit_range,
            ),
        )
        for _ in range(n_procs)
    ]

    for p in procs:
        p.start()
        time.sleep(1)
    for p in procs:
        p.join()
    pulls = np.concatenate(out_list, axis=1)

    # Plot pulls
    fig, axes = plt.subplot_mosaic(
        "AAAA\nAAAA\nAAAA\nAAAA\nAAAA\nCCDD\nCCDD\nEEFF", figsize=(8, 10)
    )
    _plot_fit(
        out_dict["fit_params"],
        out_dict["rs_count"],
        out_dict["ws_count"],
        (axes["C"], axes["E"]),
        (axes["D"], axes["F"]),
        bins,
        fit_range,
    )
    _plot_pulls(
        fig,
        axes["A"],
        pulls,
        n_procs * n_experiments,
        out_dict["labels"],
    )
    fig.tight_layout()

    path = "systematic_pull.png"
    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)

    fig.savefig(path)


if __name__ == "__main__":
    main()
