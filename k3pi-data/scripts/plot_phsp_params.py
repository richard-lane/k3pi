"""
Plot projections of and correlations between phase space
variables

"""
import sys
import pathlib
from typing import Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import phasespace as ps
from fourbody.param import helicity_param, inv_mass_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_efficiency.plotting import phsp_labels


def _gen(n_gen: int):
    """
    Generate k3pi

    """
    k_mass = 493.677
    pi_mass = 139.570
    d_mass = 1864.84
    names = ["k", "pi1", "pi2", "pi3"]
    gen = ps.nbody_decay(d_mass, (k_mass, pi_mass, pi_mass, pi_mass), names=names)

    weights, particles = gen.generate(n_gen, normalize_weights=True)
    keep = (np.max(weights) * np.random.random(n_gen)) < weights

    return tuple(particles[name].numpy()[keep].T for name in names)


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

    # Find axis parameters
    ax_x0, ax_y0, ax_width, ax_height = axis.get_position().bounds

    # Choose colourbar parameters based on these
    cbar_width = 0.05

    # Add the new axis
    fig.subplots_adjust(right=1 - cbar_width)
    cbar_ax = fig.add_axes(
        [ax_x0 + ax_width + 0.1 * cbar_width, ax_y0, cbar_width, ax_height]
    )

    # Add the colourbar
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


def _parameterise(
    k: np.ndarray, pi1: np.ndarray, pi2: np.ndarray, pi3: np.ndarray, fcn: Callable
):
    """
    Find parameterisation

    phsp points passed in, exponential times

    """
    return np.column_stack(
        (fcn(k, pi1, pi2, pi3), np.random.exponential(size=(k.shape[1])))
    )


def main():
    """
    Create plots

    """
    n_gen = 10_000_000
    k3pi = _gen(n_gen)

    helicity = _parameterise(*k3pi, helicity_param)
    mass = _parameterise(*k3pi, inv_mass_param)

    helicity_labels = phsp_labels()
    mass_labels = (
        r"$M(K^+\pi_1^-) /MeV$",
        r"$M(\pi_1^-\pi_2^-) /MeV$",
        r"$M(\pi_2^-\pi_3^+) /MeV$",
        r"$M(K^+\pi_1^-\pi_2^-) /MeV$",
        r"$M(\pi_1^-\pi_2^-\pi_3^+) /MeV$",
        r"$t / \tau$",
    )

    fig, _ = _plot(helicity, helicity_labels)
    fig.savefig("phsp_helicity.png")

    fig, _ = _plot(mass, mass_labels)
    fig.savefig("phsp_mass.png")

    fig = _plot_correlation(helicity, helicity_labels)
    fig.savefig("phsp_helicity_corr.png")

    fig = _plot_correlation(mass, mass_labels)
    fig.savefig("phsp_mass_corr.png")

    plt.show()


if __name__ == "__main__":
    main()
