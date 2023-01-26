"""
Pull study for the separate fitters

Background comes from random pions + k3pi; signal from MC
Repeat many times, see if there's any bias in the signal fraction

"""
import os
import sys
import time
import pathlib
from typing import Tuple
from multiprocessing import Manager, Process
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from lib_data import stats
from lib_data import get, training_vars
from lib_cuts.get import classifier as get_clf
from lib_cuts.definitions import THRESHOLD
from libFit import util as mass_util, fit, plotting
from libFit.definitions import mass_bins
from libFit.bkg import get_dump
from libFit.pdfs import domain


class FitNotValidException(Exception):
    """
    Fit didn't converge or something
    """


def _bkg_counts(sign: str, n_bins: int) -> np.ndarray:
    """
    Open pickle dump, return histogram of bkg counts
    after the BDT cut

    """
    return get_dump(n_bins, sign, bdt_cut=False, efficiency=False)[0]


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


def _fit(
    counts: np.ndarray, errs: np.ndarray, sign: str, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a mass fit to the provided counts and errors

    :returns: array of fit parameters
    :returns: array of fit errors

    """
    fitter = fit.alt_bkg_fit(
        counts, bins, sign, 5, 0.5, bdt_cut=False, efficiency=False, errors=errs
    )

    if not fitter.valid:
        print(fitter)

        # For plotting
        # fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB")
        # axes["A"].plot((bins[1:] + bins[:-1]) / 2, counts)
        # plotting.alt_bkg_fit(
        #     (axes["A"], axes["B"]),
        #     counts,
        #     errs,
        #     bins,
        #     fitter.values,
        #     sign=sign,
        #     bdt_cut=False,
        #     efficiency=False,
        # )
        # plt.show()
        # fig.savefig("mass_pull_invalid_fit.png")
        # plt.close(fig)

        raise FitNotValidException

    return fitter.values, fitter.errors


def _plot_pulls(sig_pull: np.ndarray, bkg_pull: np.ndarray) -> None:
    """
    Plot the pulls

    """
    fig, ax = plt.subplots(1, 2)
    hist_kw = {"histtype": "step", "bins": np.linspace(-4, 4, 30)}

    ax[0].hist(
        sig_pull,
        **hist_kw,
        label=f"{np.mean(sig_pull):.4f}" r"$\pm$" f"{np.std(sig_pull):.4f}",
    )
    ax[1].hist(
        bkg_pull,
        **hist_kw,
        label=f"{np.mean(bkg_pull):.4f}" r"$\pm$" f"{np.std(bkg_pull):.4f}",
    )

    ax[0].set_title("N sig pull")
    ax[1].set_title("N bkg pull")

    ax[0].legend()
    ax[1].legend()

    fig.suptitle(f"N={len(sig_pull)}")

    fig.tight_layout()

    fig.savefig("mass_pulls.png")
    plt.close(fig)


def _bkg_acc_rej(
    rng: np.random.Generator, bkg_pdf: np.ndarray, bins: np.ndarray
) -> np.ndarray:
    """
    Generate uniform points in unit square

    See which ones lie below the bkg_pdf counts
    Keep them

    """
    num = 1000000
    x, y = rng.random((2, num))

    # Scale x to be in the domain
    low, high = domain()
    x *= high - low
    x += low

    # Scale y to be smaller to make it more efficient
    y /= 5

    # Counts in the bin for each x value
    pdf_counts = bkg_pdf[np.digitize(x, bins) - 1]

    keep = y < pdf_counts

    # For plotting, just in case
    # plt.plot(x[keep], y[keep], "r.")
    # plt.plot(x[~keep], y[~keep], "k.")
    # plt.stairs(bkg_pdf, bins, color="g")
    # plt.show()

    count, _ = stats.counts(x[keep], bins)
    return count


def _pull(
    out_dict: dict,
    label: int,
    bkg_pdf: int,
    signal_counts: np.ndarray,
    bins: np.ndarray,
    sign: str,
) -> None:
    """
    Generate background by combining random pions with
    k3pi

    Pick a subset of the signal (MC after BDT cut) to overlay

    Find the expected number of signal and background

    Do a mass fit

    Record the pull

    stores the pull in out_dict, key starting with label

    """
    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    n_experiments = 100
    sig_pull, bkg_pull = (np.ones(n_experiments) * np.inf for _ in range(2))

    for i in tqdm(range(n_experiments)):
        # Try the fit until it converges
        while True:
            try:
                # Do accept-reject for background
                bkg_counts = _bkg_acc_rej(rng, bkg_pdf, bins)

                # Add poisson fluctuations to the signal counts
                sig_counts = rng.poisson(lam=signal_counts).astype(np.float64)

                # Scale the signal counts to give a sensible amount
                sig_scale = 0.3
                sig_counts *= sig_scale

                counts = bkg_counts + sig_counts
                errs = np.sqrt(bkg_counts + sig_scale * sig_scale * sig_counts)

                fit_params, fit_errs = _fit(counts, errs, sign, bins)

            except FitNotValidException:
                print("fit not valid")
                continue

            else:
                sig_pull[i] = (fit_params[0] - np.sum(sig_counts)) / fit_errs[0]
                bkg_pull[i] = (fit_params[1] - np.sum(bkg_counts)) / fit_errs[1]
                break

    out_dict[f"{label}_sig"] = sig_pull
    out_dict[f"{label}_bkg"] = bkg_pull


def main():
    """
    Get signal + bkg, do fits, find pulls

    Multiprocessed

    """
    # Get background counts from the pickle dump
    n_bins = 100
    year, sign, magnetisation = "2018", "dcs", "magdown"
    bkg_pdf = _bkg_counts(sign, n_bins)

    # Get MC counts from the dataframes
    bins = mass_bins(n_bins)
    sig_counts = _sig_counts(year, sign, magnetisation, bins)

    out_dict = Manager().dict()
    n_procs = 6
    procs = [
        Process(target=_pull, args=(out_dict, i, bkg_pdf, sig_counts, bins, sign))
        for i in range(n_procs)
    ]

    for p in procs:
        p.start()
        time.sleep(1)
    for p in procs:
        p.join()

    sig_pull = np.concatenate(
        [item for key, item in out_dict.items() if key.endswith("_sig")]
    )

    bkg_pull = np.concatenate(
        [item for key, item in out_dict.items() if key.endswith("_bkg")]
    )

    _plot_pulls(sig_pull, bkg_pull)


if __name__ == "__main__":
    main()
