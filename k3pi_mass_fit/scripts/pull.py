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
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))

from lib_cuts.get import classifier as get_clf, cut_dfs
from lib_cuts.definitions import THRESHOLD
from lib_efficiency.efficiency_util import k_3pi
from lib_data import stats
from lib_data import get, definitions, util, training_vars
from libFit.definitions import mass_bins
from libFit import util as mass_util, fit


def _gen_bkg(
    i: int, year: str, sign: str, magnetisation: str, bins: np.ndarray
) -> np.ndarray:
    """
    Combine slow pions from different events
    with the K3pi to find background

    :param i: tells us which slow pion to use
    :returns: array of background delta masses

    """
    data_generator = get.data(year, sign, magnetisation)

    clf = get_clf(year, "dcs", magnetisation)
    data_generator = cut_dfs(data_generator, clf)

    masses = []

    for dataframe in data_generator:
        k3pi = k_3pi(dataframe)

        slowpi = np.row_stack(
            [dataframe[f"slowpi_{s}"] for s in definitions.MOMENTUM_SUFFICES]
        )
        d_mass = util.inv_mass(*k3pi)

        slowpi = np.roll(slowpi, i + 1, axis=1)

        dst_mass = util.inv_mass(*k3pi, slowpi)
        delta_m = dst_mass - d_mass
        keep = (delta_m > bins[0]) & (delta_m < bins[-1])

        masses.append(delta_m[keep])

    return np.concatenate(masses)


def _select_sig(
    gen: np.random.Generator, fraction: float, year: str, sign: str, magnetisation: str
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

    # Randomly choose some of it
    mask = gen.random(len(mc_df)) < fraction
    return mass_util.delta_m(mc_df)[mask]


def _count_err(
    bkg: np.ndarray, sig: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin invariant masses

    :returns: counts in each bin
    :returns: error on counts in each bin

    """
    return stats.counts(np.concatenate((bkg, sig)), bins)


def _fit(
    counts: np.ndarray, errs: np.ndarray, sign: str, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a mass fit to the provided counts and errors

    :returns: array of fit parameters
    :returns: array of fit errors

    """
    fitter = fit.alt_bkg_fit(
        counts, bins, sign, 5, 0.5, errors=errs, bdt_cut=True, efficiency=False
    )

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

    fig.tight_layout()

    fig.savefig("mass_pulls.png")
    plt.close(fig)


def _pull(out_dict: dict, label: int):
    """
    Generate background by combining random pions with
    k3pi

    Pick a subset of the signal (MC after BDT cut) to overlay

    Find the expected number of signal and background

    Do a mass fit

    Record the pull

    stores the pull in out_dict, with key starting with i

    """
    n_experiments = 10
    sig_pull, bkg_pull = (np.ones(n_experiments) * np.inf for _ in range(2))

    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    year, sign, magnetisation = "2018", "cf", "magdown"

    # How much of the MC dataframe we use for the signal
    sig_proportion = 0.03

    bins = mass_bins(100)

    for i in tqdm(range(n_experiments)):
        bkg = _gen_bkg(i, year, sign, magnetisation, bins)
        sig = _select_sig(rng, sig_proportion, year, sign, magnetisation)

        counts, errs = _count_err(bkg, sig, bins)
        # centres = (bins[1:] + bins[:-1]) / 2
        # widths = (bins[1:] - bins[:-1]) / 2
        # plt.errorbar(centres, counts, xerr=widths, yerr=errs, fmt="k+")

        fit_params, fit_errs = _fit(counts, errs, sign, bins)

        sig_pull[i] = (fit_params[0] - len(sig)) / fit_errs[0]
        bkg_pull[i] = (fit_params[1] - len(bkg)) / fit_errs[1]

    out_dict[f"{label}_sig"] = sig_pull
    out_dict[f"{label}_bkg"] = bkg_pull


def main():
    """
    Get signal + bkg, do fits, find pulls

    Multiprocessed

    """
    out_dict = Manager().dict()
    n_procs = 6

    procs = [Process(target=_pull, args=(out_dict, i)) for i in range(n_procs)]

    for p in procs:
        p.start()
        time.sleep(1)
    for p in procs:
        p.join()

    sig_pull = np.concatenate(
        [out_dict[key] for key in out_dict.keys() if key.endswith("_sig")]
    )

    bkg_pull = np.concatenate(
        [item for key, item in out_dict.items() if key.endswith("_bkg")]
    )

    _plot_pulls(sig_pull, bkg_pull)


if __name__ == "__main__":
    main()
