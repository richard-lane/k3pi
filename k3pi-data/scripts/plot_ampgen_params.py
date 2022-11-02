"""
Plot projections of and correlations between phase space
variables for ampgen

"""
import sys
import pathlib
from typing import Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param, inv_mass_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_data import get, util
from lib_efficiency import efficiency_util
from lib_efficiency.plotting import phsp_labels


def _plot_correlation(points: np.ndarray, labels) -> plt.Figure:
    """
    Plot correlation matrix

    """
    fig, axis = plt.subplots()
    num = len(labels)
    corr = np.ones((num, num)) * np.inf

    for i in range(num):
        for j in range(num):
            corr[i, j] = np.corrcoef(points[:, i], points[:, j])[0, 1]

    plt.set_cmap("seismic")
    axis.imshow(corr, vmin=-1.0, vmax=1.0)

    axis.set_xticks(range(num))
    axis.set_yticks(range(num))

    axis.set_xticklabels(labels, rotation=90)
    axis.set_yticklabels(labels)

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.30, 0.05, 0.50])

    fig.colorbar(mappable=axis.get_images()[0], cax=cbar_ax)

    return fig


def _plot(points: np.ndarray, labels) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of points and labels on an axis; return the figure and axis

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for axis, point, label in zip(ax.ravel(), points.T, labels):
        contents, _, _ = axis.hist(point, bins=100, **hist_kw)

        # No idea why I have to do this manually
        axis.set_ylim(0, np.max(contents) * 1.1)
        axis.set_yticks([])

        axis.set_title(label)
    fig.tight_layout()

    return fig, ax


def _parameterise(data_frame: pd.DataFrame, fcn: Callable):
    """
    Find parameterisation of a dataframe

    """
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(data_frame)
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    return np.column_stack((fcn(k, pi1, pi2, pi3), data_frame["time"]))


def main():
    """
    Create plots

    """
    ampgen_df = get.ampgen("dcs")
    ampgen_helicity = _parameterise(ampgen_df, helicity_param)
    ampgen_mass = _parameterise(ampgen_df, inv_mass_param)

    helicity_labels = phsp_labels()
    mass_labels = (
        r"$M(K^+\pi_1^-) /MeV$",
        r"$M(\pi_1^-\pi_2^-) /MeV$",
        r"$M(\pi_2^-\pi_3^+) /MeV$",
        r"$M(K^+\pi_1^-\pi_2^-) /MeV$",
        r"$M(\pi_1^-\pi_2^-\pi_3^+) /MeV$",
        r"$t / \tau$",
    )

    fig, _ = _plot(ampgen_helicity, helicity_labels)
    fig.savefig("ampgen_helicity.png")

    fig, _ = _plot(ampgen_mass, mass_labels)
    fig.savefig("ampgen_mass.png")

    fig = _plot_correlation(ampgen_helicity, helicity_labels)
    fig.savefig("ampgen_helicity_corr.png")

    fig = _plot_correlation(ampgen_mass, mass_labels)
    fig.savefig("ampgen_mass_corr.png")

    plt.show()


if __name__ == "__main__":
    main()
