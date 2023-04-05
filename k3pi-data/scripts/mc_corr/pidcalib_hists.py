"""
Plot histograms of PID calibration efficiences,
generated with pidcalib2.

If you haven't created these already, create them
with the script
k3pi-data/scripts/create_pid_hists.sh

"""
import os
import sys
import pickle
import pathlib
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import corrections


def main(
    *,
    year: str,
    magnetisation: str,
):
    """
    Read + plot histograms

    """
    k_hist, pi_hist = corrections.k_pi_hists(year, magnetisation)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    hist_options = {"vmin": 0, "vmax": 1, "cmap": "magma"}
    k_mesh = axes[0].pcolormesh(
        *k_hist.axes.edges.T, k_hist.values().T, vmin=0, vmax=1, cmap="magma"
    )
    axes[1].pcolormesh(
        *pi_hist.axes.edges.T, pi_hist.values().T, vmin=0, vmax=1, cmap="magma"
    )

    axes[0].set_title(r"K")
    axes[1].set_title(r"$\pi$")

    for axis in axes:
        axis.set_xlabel(r"$p$")
        axis.set_ylabel(r"$\eta$")

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.90, 0.1, 0.05, 0.8])
    fig.colorbar(k_mesh, cax=cax)

    path = f"pidcalib_hists_{year}_{magnetisation}.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pidcalib2 histograms")
    parser.add_argument(
        "year", type=str, choices={"2016", "2017", "2018"}, help="Data taking year"
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="Magnetisation Direction",
    )

    main(**vars(parser.parse_args()))
