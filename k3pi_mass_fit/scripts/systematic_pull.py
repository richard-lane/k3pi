"""
Pull study to assess systematic associated with the mass fitter

"""
import os
import sys
import time
import pathlib
from multiprocessing import Process, Manager
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from lib_data import stats
from lib_data import get, training_vars
from lib_cuts.get import classifier as get_clf
from lib_cuts.definitions import THRESHOLD
from libFit import util as mass_util, fit, plotting, pdfs, definitions
from libFit.bkg import get_dump


class GaussKDE:
    """
    Callable for KDE

    Unused
    """

    def __init__(self, positions: np.ndarray, counts: np.ndarray, width: float):
        """
        Tell us where to put the Gaussians and how strong to make them
        """
        self._gaussians = (norm(loc=position, scale=width) for position in positions)
        self._counts = counts

    def __call__(self, points: np.ndarray):
        """
        Evaluate the estimated PDF

        """
        retval = np.zeros_like(points)

        # Could vectorise
        for gaussian, count in zip(self._gaussians, self._counts):
            retval += count * gaussian.pdf(points)

        retval /= np.sum(self._counts)

        return retval


def _bkg_pdf(sign: str, n_bins: int) -> Callable:
    """
    Open pickle dump, return histogram of bkg counts
    after the BDT cut

    Gaussian smooths the counts as well TODO

    """
    bins = definitions.mass_bins(n_bins)
    centres = (bins[1:] + bins[:-1]) / 2

    # TODO BDT cut should be false, but it's very jagged
    counts, _ = get_dump(n_bins, sign, bdt_cut=False, efficiency=False)

    return UnivariateSpline(centres, counts, k=4, s=5.0)


def _sig_counts(
    year: str, sign: str, magnetisation: str, bins: np.ndarray
) -> np.ndarray:
    """
    Randomly choose a subset of the MC

    :returns: array of signal delta masses

    """
    # Read the right dataframe
    mc_df = get.mc(year, sign, magnetisation)

    # Perform BDT cut
    clf = get_clf(year, "dcs", magnetisation)

    training_labels = list(training_vars.training_var_names())
    sig_predictions = clf.predict_proba(mc_df[training_labels])[:, 1] > THRESHOLD

    mc_df = mc_df[sig_predictions]

    count, _ = stats.counts(mass_util.delta_m(mc_df), bins)

    return count


def _bkg(
    rng, bkg_pdf: np.ndarray, n_gen: int, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Background counts with Poisson errors

    """
    # Accept reject sample
    low, high = pdfs.domain()
    max_ = 0.11

    x = low + (high - low) * rng.random(n_gen)
    y = max_ * rng.random(n_gen)

    assert (bkg_pdf(x) < max_).all()

    points = x[y < bkg_pdf(x)]

    counts, _ = stats.counts(points, bins)
    return counts


def _sig(rng: np.random.Generator, sig_counts: np.ndarray, n_tot: int) -> np.ndarray:
    """
    Array of signal counts after Poisson fluctuations

    """
    # Scale such that we have the right counts
    counts = sig_counts * n_tot / np.sum(sig_counts)

    # Apply Poisson fluctuations
    return rng.poisson(lam=counts).astype(np.float64)


def _pull_study(
    out_list: list,
    out_dict: dict,
    bkg_pdf: int,
    rs_signal_counts: np.ndarray,
    ws_signal_counts: np.ndarray,
    bins: np.ndarray,
):
    """
    Generate bkg using accept-reject

    Find signal counts from MC and apply Poisson fluctuations

    """
    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    n_rs_sig = 800_000
    n_ws_sig = 2_200

    n_experiments = 25
    # want to track n_sig and n_bkg for both RS and WS
    pulls = [np.full(n_experiments, np.inf, dtype=float) for _ in range(4)]
    for i in tqdm(range(n_experiments)):
        while True:
            # Generate bkg
            rs_bkg = _bkg(rng, bkg_pdf, 45_000, bins=bins)
            ws_bkg = _bkg(rng, bkg_pdf, 45_000, bins=bins)

            # Add fluctuations to sig
            rs_sig = _sig(rng, rs_signal_counts, n_rs_sig)
            ws_sig = _sig(rng, ws_signal_counts, n_ws_sig)

            # Combine
            rs_counts = rs_sig + rs_bkg
            ws_counts = ws_sig + ws_bkg

            # Do a fit
            binned_fitter = fit.binned_simultaneous_fit(
                rs_counts,
                ws_counts,
                bins,
                5,
            )
            if not binned_fitter.valid:
                print("fit not valid")
                continue

            # Find the pulls
            fit_vals = binned_fitter.values
            fit_errs = binned_fitter.errors
            rs_sig_pull = (fit_vals[0] - n_rs_sig) / fit_errs[0]
            rs_bkg_pull = (fit_vals[1] - np.sum(rs_bkg)) / fit_errs[1]
            ws_sig_pull = (fit_vals[2] - n_ws_sig) / fit_errs[2]
            ws_bkg_pull = (fit_vals[3] - np.sum(ws_bkg)) / fit_errs[3]

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


def _plot_fit(fit_params, rs_count, ws_count, rs_fit_axes, ws_fit_axes, bins):
    """
    Plot fit on an axis

    """
    for axes, counts, params in zip(
        (rs_fit_axes, ws_fit_axes),
        (rs_count, ws_count),
        mass_util.rs_ws_params(fit_params),
    ):
        plotting.mass_fit(axes, counts, np.sqrt(counts), bins, params)


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

    pull_axis.set_xlim(-10.0, 10.0)

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
    # Get background PDF from the pickle dump
    bkg_pdf = _bkg_pdf("dcs", n_bins=100)

    # Get MC counts from the dataframes
    year, magnetisation = "2018", "magdown"
    bins = np.unique(
        np.concatenate(
            (
                np.linspace(pdfs.domain()[0], 144.5, 75),
                np.linspace(144.5, 146.5, 175),
                np.linspace(146.5, pdfs.domain()[1], 75),
            )
        )
    )

    rs_count = _sig_counts(year, "cf", magnetisation, bins)
    ws_count = _sig_counts(year, "dcs", magnetisation, bins)

    manager = Manager()
    out_dict = manager.dict()
    out_list = manager.list()
    n_procs = 6
    procs = [
        Process(
            target=_pull_study,
            args=(out_list, out_dict, bkg_pdf, rs_count, ws_count, bins),
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
        "AAAA\nAAAA\nAAAA\nAAAA\nAAAA\nCCDD\nCCDD\nEEFF\nEEFF", figsize=(8, 10)
    )
    _plot_fit(
        out_dict["fit_params"],
        out_dict["rs_count"],
        out_dict["ws_count"],
        (axes["C"], axes["E"]),
        (axes["D"], axes["F"]),
        bins,
    )
    _plot_pulls(
        fig,
        axes["A"],
        pulls,
        n_procs * len(pulls.T[0]),
        out_dict["labels"],
    )
    fig.tight_layout()

    fig.savefig("systematic_pull.png")


if __name__ == "__main__":
    main()
