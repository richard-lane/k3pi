"""
Plot ampgen parameterisations

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


def _plot(points: np.ndarray, names: Tuple[str]) -> plt.Figure:
    """
    Plot parameterisation from an (N, 6) arrray of points on an axis

    Returns figure

    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    hist_kw = {"density": True, "histtype": "step"}
    for axis, data, label in zip(axes.ravel(), points.T, names):
        contents, _, _ = axis.hist(data, bins=100, **hist_kw)
        axis.set_ylim(0, np.max(contents) * 1.1)

        axis.set_xlabel(label)
        axis.set_yticks([])

    fig.tight_layout()

    return fig


def _plot_cm(dataframe: pd.DataFrame):
    """
    Plot correlation the CM param of a dataframe

    """
    cm_pts = np.column_stack((helicity_param(*k_3pi(dataframe)), dataframe["time"]))

    fig = _plot(cm_pts, phsp_labels())

    fig.savefig("ampgen_param_cm.png")
    plt.close(fig)


def _plot_invmass(dataframe: pd.DataFrame):
    """
    Plot correlation the CM param of a dataframe

    """
    mass_pts = np.column_stack((inv_mass_param(*k_3pi(dataframe)), dataframe["time"]))

    # Plot matrices
    fig = _plot(mass_pts, (
        r"$M(K^+\pi_1^-)$",
        r"$M(\pi_1^-\pi_2^-)$",
        r"$M(\pi_2^-\pi^+)$",
        r"$M(K^+\pi_1^-\pi_2^-)$",
        r"$M(\pi_1^-\pi_2^-\pi^+)$",
        r"t / $\tau$"
        )
        )

    fig.savefig("ampgen_param_mass.png")
    plt.close(fig)


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
