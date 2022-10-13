"""
Reweight MC using the weights found in the pidcalib histograms

"""
import sys
import pickle
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import corrections, get, util


def _get_hists() -> Tuple[bh.Histogram, bh.Histogram]:
    """
    Read pi and K histograms from pickle dumps

    """
    pi_hist, k_hist = None, None
    paths = tuple(pathlib.Path("pidcalib_output/").glob("*"))

    for path in paths:
        if not pi_hist and "-Pi-" in str(path):
            print(f"Pi: {path}")
            with open(path, "rb") as pi_f:
                pi_hist = pickle.load(pi_f)

        elif not k_hist and "-K-" in str(path):
            print(f"K: {path}")
            with open(path, "rb") as k_f:
                k_hist = pickle.load(k_f)

        else:
            print(f"not plotting {path} (neither k nor pi?)")

    return pi_hist, k_hist


def _eta_and_momenta(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get K3pi eta and momentum from a dataframe

    """
    k_eta = util.eta(
        dataframe["Kplus_Px"], dataframe["Kplus_Py"], dataframe["Kplus_Pz"]
    )
    pi_eta = tuple(
        util.eta(dataframe[f"pi{s}_Px"], dataframe[f"pi{s}_Py"], dataframe[f"pi{s}_Pz"])
        for s in ("1minus", "2minus", "3plus")
    )
    etas = np.row_stack((k_eta, *pi_eta))

    k_p = np.sqrt(
        dataframe["Kplus_Px"] ** 2
        + dataframe["Kplus_Py"] ** 2
        + dataframe["Kplus_Pz"] ** 2
    )
    pi_p = tuple(
        np.sqrt(
            dataframe[f"pi{s}_Px"] ** 2
            + dataframe[f"pi{s}_Py"] ** 2
            + dataframe[f"pi{s}_Pz"] ** 2
        )
        for s in ("1minus", "2minus", "3plus")
    )
    momenta = np.row_stack((k_p, *pi_p))

    return etas, momenta


def main():
    """
    Read MC, find the right weights, plot histograms before/after reweighting

    """
    # Get the pidcalib histograms
    pi_hist, k_hist = _get_hists()

    # Get the MC dataframe
    mc_df = get.mc("2018", "dcs", "magdown")
    mc_etas, mc_ps = _eta_and_momenta(mc_df)
    wt = corrections.pid_weights(mc_etas, mc_ps, pi_hist, k_hist)

    # Get real data
    data_etas, data_ps = _eta_and_momenta(pd.concat(get.data("2018", "dcs", "magdown")))

    # plot
    fig, ax = plt.subplots(4, 2, figsize=(14, 7))
    kw = {"density": False, "histtype": "step"}

    # Weight real data so the histograms look more sensible
    data_wt = np.ones_like(data_etas[0]) * len(mc_etas[0]) / len(data_etas[0])

    for mc_eta, data_eta, axis, label in zip(
        mc_etas, data_etas, ax[:, 0], (r"$K^+$", r"$\pi^-$", r"$\pi^+$", r"$\pi^-$")
    ):
        bins = np.linspace(1.5, 5.5, 100)
        axis.hist(mc_eta, **kw, bins=bins, label="MC unweighted")
        axis.hist(mc_eta, **kw, bins=bins, weights=wt, label="MC PIDcalib Weighted")
        axis.hist(data_eta, **kw, bins=bins, weights=data_wt, label="Data")
        axis.set_ylabel(label)

    for mc_p, data_p, axis in zip(mc_ps, data_ps, ax[:, 1]):
        bins = np.linspace(0, 100000, 100)
        axis.hist(mc_p, **kw, bins=bins, label="MC unweighted")
        axis.hist(mc_p, **kw, bins=bins, weights=wt, label="MC PIDcalib Weighted")
        axis.hist(data_p, **kw, bins=bins, weights=data_wt, label="Data")

    for axis in ax.ravel():
        axis.set_yticks([])

    ax[0, 1].legend()
    ax[0, 0].set_title(r"$\eta$")
    ax[0, 1].set_title(r"p / MeV")

    fig.tight_layout()

    fig.savefig("pid_reweight.png")
    plt.show()


if __name__ == "__main__":
    main()
