"""
Reweight MC using the weights found in the pidcalib histograms

"""
import sys
import pickle
import pathlib
from typing import Tuple
import numpy as np
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


def main():
    """
    Read MC, find the right weights, plot histograms before/after reweighting

    """
    # Get the pidcalib histograms
    pi_hist, k_hist = _get_hists()

    # Get the MC dataframe
    mc_df = get.mc("2018", "dcs", "magdown")
    kw = {"bins": np.linspace(1.5, 5.2, 300), "histtype": "step"}
    k_eta = util.eta(mc_df["Kplus_Px"], mc_df["Kplus_Py"], mc_df["Kplus_Pz"])
    pi_eta = tuple(
        util.eta(mc_df[f"pi{s}_Px"], mc_df[f"pi{s}_Py"], mc_df[f"pi{s}_Pz"])
        for s in ("1minus", "2minus", "3plus")
    )
    etas = np.row_stack((k_eta, *pi_eta))

    k_p = np.sqrt(
        mc_df["Kplus_Px"] ** 2 + mc_df["Kplus_Py"] ** 2 + mc_df["Kplus_Pz"] ** 2
    )
    pi_p = tuple(
        np.sqrt(
            mc_df[f"pi{s}_Px"] ** 2 + mc_df[f"pi{s}_Py"] ** 2 + mc_df[f"pi{s}_Pz"] ** 2
        )
        for s in ("1minus", "2minus", "3plus")
    )
    momenta = np.row_stack((k_p, *pi_p))

    wt = corrections.pid_weights(etas, momenta, pi_hist, k_hist)

    fig, ax = plt.subplots(4, 2, figsize=(10, 5))
    kw = {"density": False, "histtype": "step"}

    for eta, axis, label in zip(etas, ax[:, 0], (r"$K^+$", r"$\pi^-$", r"$\pi^+$", r"$\pi^-$")):
        bins = np.linspace(1.5, 5.5, 100)
        axis.hist(eta, **kw, bins=bins)
        axis.hist(eta, **kw, bins=bins, weights=wt, label="Weighted")
        axis.set_ylabel(label)

    for momentum, axis in zip(momenta, ax[:, 1]):
        bins = np.linspace(0, 100000, 100)
        axis.hist(momentum, **kw, bins=bins)
        axis.hist(momentum, **kw, bins=bins, weights=wt, label="Weighted")

    for axis in ax.ravel():
        axis.set_yticks([])

    ax.ravel()[0].legend()
    ax[0, 0].set_title(r"$\eta$")
    ax[0, 1].set_title(r"p")

    fig.tight_layout()

    fig.savefig("pid_reweight.png")
    plt.show()


if __name__ == "__main__":
    main()
