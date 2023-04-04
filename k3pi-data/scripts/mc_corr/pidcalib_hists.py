"""
Plot histograms of PID calibration efficiences,
generated with pidcalib2.

If you haven't created these already, create them
with the script
k3pi-data/scripts/create_pid_hists.sh

"""
import os
import pickle
import pathlib
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh


def _plot(axis: plt.Axes, path: str) -> QuadMesh:
    """
    Plot histogram stored in a pickle dump on an axis

    """
    with open(path, "rb") as hist_f:
        hist = pickle.load(hist_f)
        return axis.pcolormesh(
            *hist.axes.edges.T, hist.values().T, vmin=0, vmax=1, cmap="magma"
        )


def main(
    *,
    year: str,
    magnetisation: str,
):
    """
    Read + plot histograms

    """
    k_path, pi_path = (
        f"pidcalib_output/effhists-Turbo{year[-2:]}-{magnetisation[3:]}-{particle}-probe_PIDK{condition}-P.ETA.pkl"
        for particle, condition in zip(("K", "Pi"), (">8", "<0"))
    )

    assert os.path.exists(k_path)
    assert os.path.exists(pi_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    pi_mesh = _plot(axes[0], pi_path)
    k_mesh = _plot(axes[1], k_path)

    axes[0].set_title(r"$\pi$")
    axes[1].set_title(r"K")

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
