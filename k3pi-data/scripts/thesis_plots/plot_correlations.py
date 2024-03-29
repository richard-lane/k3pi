"""
Plot correlation matrices for the parameterisations

"""
import sys
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param, inv_mass_param

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_efficiency"))

from lib_data import get
from lib_data.util import k_3pi
from lib_efficiency.plotting import phsp_labels


def _add_cbar(fig: plt.Figure, axis: plt.Axes) -> None:
    """
    Add colour bar to figure, using the provided axis as scale

    """
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.30, 0.05, 0.50])

    fig.colorbar(mappable=axis.get_images()[0], cax=cbar_ax)


def _plot(axis: plt.Axes, points: np.ndarray, names: Tuple[str]) -> None:
    """
    Plot correlation matrix from an (N, 6) arrray of points on an axis

    """
    num = len(names)
    corr = np.ones((num, num))

    for i, _ in enumerate(names):
        for j, _ in enumerate(names):
            corr[i, j] = np.corrcoef(points[:, i], points[:, j])[0, 1]

    axis.imshow(corr, vmin=-1.0, vmax=1.0, cmap="seismic")

    axis.set_xticks(range(num))
    axis.set_yticks(range(num))

    axis.set_xticklabels(names, rotation=90)
    axis.set_yticklabels(names)


def _plot_cm(dataframe: pd.DataFrame):
    """
    Plot correlation the CM param of a dataframe

    """
    fig, axis = plt.subplots()

    # Get CM param
    cm_pts = np.column_stack((helicity_param(*k_3pi(dataframe)), dataframe["time"]))

    # Plot matrices
    _plot(axis, cm_pts, phsp_labels())

    fig.tight_layout()

    # Plot colour bars
    _add_cbar(fig, axis)

    fig.savefig("ampgen_corr_cm.png")


def _plot_invmass(dataframe: pd.DataFrame):
    """
    Plot correlation the CM param of a dataframe

    """
    fig, axis = plt.subplots()

    # Get CM param
    mass_pts = np.column_stack((inv_mass_param(*k_3pi(dataframe)), dataframe["time"]))

    # Plot matrices
    _plot(
        axis,
        mass_pts,
        (
            r"$M(K^+\pi_1^-)$",
            r"$M(\pi_1^-\pi_2^-)$",
            r"$M(\pi_2^-\pi^+)$",
            r"$M(K^+\pi_1^-\pi_2^-)$",
            r"$M(\pi_1^-\pi_2^-\pi^+)$",
            r"t / $\tau$",
        ),
    )

    fig.tight_layout()
    # Plot colour bars
    _add_cbar(fig, axis)

    fig.savefig("ampgen_corr_mass.png")

    plt.show()


def main():
    """
    Plot correlation matrices

    """
    # Get the AmpGen dataframe
    dataframe = get.ampgen("dcs")

    _plot_cm(dataframe)
    _plot_invmass(dataframe)


if __name__ == "__main__":
    main()
