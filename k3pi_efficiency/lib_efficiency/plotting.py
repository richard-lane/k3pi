"""
Functions for plotting things

"""
import sys
import pathlib
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

from . import phsp_binning

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import stats


def phsp_labels() -> Tuple:
    """
    Plot labels for phase space variables and time

    """
    return (
        r"$M(K^+\pi^+) /MeV$",
        r"$M(\pi_1^-\pi_2^-) /MeV$",
        r"cos($\theta_+$)",
        r"cos($\theta_-$)",
        r"$\phi$",
        r"t / $\tau$",
    )


def projections(
    mc: np.ndarray, ag: np.ndarray, mc_wt: np.ndarray = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot projections of 6d points

    :param mc: (Nx6) shape array of monte carlo (i.e. the original distribution) points
    :param ag: (Nx6) shape array of AmpGen (i.e. the target distribution) points
    :param mc_wt: weights to apply to the MC points
    :param mc_wt: weights to apply to the AmpGen points

    :returns: the figure and axes that we have plotted on

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    hist_kw = {"density": True, "histtype": "step"}
    for axis, ag_x, mc_x, label in zip(ax.ravel(), ag.T, mc.T, phsp_labels()):
        contents, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw, alpha=0.5)
        if mc_wt is not None:
            axis.hist(mc_x, bins=bins, label="Reweighted", **hist_kw, weights=mc_wt)
        axis.set_ylim(0, np.max(contents) * 1.1)

        axis.set_xlabel(label)

    ax[0, 0].legend()
    fig.tight_layout()

    return fig, ax


def _plot_shaded_area(
    ax: plt.Axes, opt: np.ndarray, cov: np.ndarray, pts: np.ndarray
) -> None:
    """Plot shaded area around the best fit on the axis"""
    err = np.sqrt(np.diag(cov))
    a_max, b_max = opt[0] + err[0], opt[1] + err[1]
    a_min, b_min = opt[0] - err[0], opt[1] - err[1]

    ax.fill_between(
        pts,
        [a_max * x + b_max for x in pts],
        [a_min * x + b_min for x in pts],
        color="r",
        alpha=0.2,
    )

    # Display gradient on the plot
    # ax.text(2.0, 0.85, rf"{opt[0]:.4f}$\pm${err[0]:.4f}")


def _plot_ratio(
    ax: plt.Axes,
    numerator: np.ndarray,
    denominator: np.ndarray,
    num_wt: np.ndarray,
    denom_wt: np.ndarray,
    bins: np.ndarray,
) -> float:
    """
    Plot the ratio of two histograms on an axis

    Returns chi2

    """
    # Just use the centre of each bin for plotting
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    num_counts, num_errs = stats.counts(numerator, bins, num_wt)
    denom_counts, denom_errs = stats.counts(denominator, bins, denom_wt)

    # Scale
    scale_factor = np.sum(num_counts) / np.sum(denom_counts)
    denom_counts *= scale_factor
    denom_errs *= scale_factor

    ratio = num_counts / denom_counts
    err = ratio * np.sqrt(
        (num_errs / num_counts) ** 2 + (denom_errs / denom_counts) ** 2
    )

    # Plot
    ax.errorbar(
        centres,
        ratio,
        xerr=widths,
        yerr=err,
        fmt="k+",
    )
    ax.set_ylim(0.8, 1.2)

    # Plot ideal
    pts = np.linspace(*ax.get_xlim())
    ax.plot(pts, np.ones_like(pts), "k--")

    # Do fit, plot best
    opt, cov = curve_fit(
        lambda x, a, b: a * x + b,
        centres,
        ratio,
        p0=(0, 1),
        sigma=err,
        absolute_sigma=True,
    )

    def best_fit(t: float) -> float:
        """Best fit fcn"""
        return opt[0] * t + opt[1]

    ax.plot(pts, [best_fit(x) for x in pts], "r--")

    # Plot also shaded area
    _plot_shaded_area(ax, opt, cov, pts)

    # Find chi2
    n_dof = (len(bins) - 1) - 2  # n_bins - 2 since we have 2 fit params
    return np.sum(((ratio - best_fit(centres)) ** 2) / (n_dof * err**2))


def plot_ratios(
    rs_mc_t, ws_mc_t, rs_ag_t, ws_ag_t, rs_mc_wt, ws_mc_wt, bins: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot WS/RS ratios for mc, ampgen, reweighted

    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    mc_chi2 = _plot_ratio(
        ax[0], ws_mc_t, rs_mc_t, np.ones_like(ws_mc_t), np.ones_like(rs_mc_t), bins
    )
    ag_chi2 = _plot_ratio(
        ax[1], ws_ag_t, rs_ag_t, np.ones_like(ws_ag_t), np.ones_like(rs_ag_t), bins
    )
    reweighted_chi2 = _plot_ratio(ax[2], ws_mc_t, rs_mc_t, ws_mc_wt, rs_mc_wt, bins)

    ax[0].set_title(rf"MC, $\chi^2=${mc_chi2:.3f}")
    ax[1].set_title(rf"AmpGen, $\chi^2=${ag_chi2:.3f}")
    ax[2].set_title(rf"Reweighted, $\chi^2=${reweighted_chi2:.3f}")

    ax[0].set_ylabel(r"$\frac{WS}{RS}$ ratio")
    for a in ax:
        a.set_xlabel(r"t/$\tau$")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0)

    return fig, ax


def _chunks_z(
    k: np.ndarray,
    pi1: np.ndarray,
    pi2: np.ndarray,
    pi3: np.ndarray,
    n: int,
    weights: np.ndarray = None,
) -> List[Tuple[float, float]]:
    """
    Split our arrays into n approx. equal chunks, evaluate Z for each chunk

    :param k: (4, N) numpy array of k (px, py, pz, E). Assumes K+
    :param pi1: (4, N) numpy array of pi1 (px, py, pz, E). Assumes pi-
    :param pi2: (4, N) numpy array of pi2 (px, py, pz, E). Assumes pi-
    :param pi3: (4, N) numpy array of pi3 (px, py, pz, E). Assumes pi+
    :param n: number of chunks to spit our arrays into
    :param weights: weights to apply when evaluating coherence factors

    :returns: an n-length list of the coherence factor as (real, imag) for each chunk

    """
    if weights is None:
        weights = np.ones(len(k[0]))

    split_k = np.array_split(k, n, axis=1)
    split_pi1 = np.array_split(pi1, n, axis=1)
    split_pi2 = np.array_split(pi2, n, axis=1)
    split_pi3 = np.array_split(pi3, n, axis=1)
    split_weights = np.array_split(weights, n)

    def z(mag: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        return re, im from mag and phase

        """
        return mag * np.cos(phase * np.pi / 180.0), mag * np.sin(phase * np.pi / 180.0)

    points = []
    for i in range(n):
        this_z = phsp_binning.coherence_factor(
            split_k[i],
            split_pi1[i],
            split_pi2[i],
            split_pi3[i],
            weights=split_weights[i],
        )
        points.append(z(*this_z))

    return points


def _plot_hull(
    ax: plt.Axes, points: List[Tuple[float, float]], color: str
) -> ConvexHull:
    """
    Plot the convex hull surrounding a set of points on an axis

    """
    # Plot hulls
    hull = ConvexHull(points)
    array_points = np.array(points)
    for s in hull.simplices:
        ax.plot(array_points[s, 0], array_points[s, 1], f"{color}--", alpha=0.5)

    return hull


def z_scatter(
    target_k: np.ndarray,
    target_pi1: np.ndarray,
    target_pi2: np.ndarray,
    target_pi3: np.ndarray,
    orig_k: np.ndarray,
    orig_pi1: np.ndarray,
    orig_pi2: np.ndarray,
    orig_pi3: np.ndarray,
    orig_wt: np.ndarray,
    n: int,
    default_scale: bool = True,
) -> Tuple[plt.Figure, plt.Axes, float, float]:
    """
    Plot a scatter plot of the numerically evaluated coherence factor for a target dataset,
    an original dataset and a weighted version of the original dataset.

    Splits data into N chunks to get an idea of the statistical error involved

    :param target_k: (4, N) numpy array of k (px, py, pz, E). Assumes K+
    :param target_pi1: (4, N) numpy array of pi1 (px, py, pz, E). Assumes pi-
    :param target_pi2: (4, N) numpy array of pi2 (px, py, pz, E). Assumes pi-
    :param target_pi3: (4, N) numpy array of pi3 (px, py, pz, E). Assumes pi+
    :param orig_k: (4, N) numpy array of k (px, py, pz, E). Assumes K+
    :param orig_pi1: (4, N) numpy array of pi1 (px, py, pz, E). Assumes pi-
    :param orig_pi2: (4, N) numpy array of pi2 (px, py, pz, E). Assumes pi-
    :param orig_pi3: (4, N) numpy array of pi3 (px, py, pz, E). Assumes pi+
    :param orig_wt: weights to apply to the original dataset
    :param n: number of chunks to spit our arrays into
    :param default_scale: whether to use matplotlib's default scaling (True), or to zoom out
                          and plot both axes from -1 to +1
    :returns: the figure used to plot on
    :returns: the axis used to plot on
    :returns: distance between ampgen and reweighted hulls
    :returns: area of reweighted hull

    """

    fig, ax = plt.subplots()

    orig_points = _chunks_z(orig_k, orig_pi1, orig_pi2, orig_pi3, n)
    target_points = _chunks_z(target_k, target_pi1, target_pi2, target_pi3, n)
    reweighted_points = _chunks_z(orig_k, orig_pi1, orig_pi2, orig_pi3, n, orig_wt)

    # Plot stuff
    ax.plot([x[0] for x in orig_points], [x[1] for x in orig_points], "+", color="r")
    ax.plot(
        [x[0] for x in target_points], [x[1] for x in target_points], "+", color="g"
    )
    ax.plot(
        [x[0] for x in reweighted_points],
        [x[1] for x in reweighted_points],
        "+",
        color="b",
    )

    # Plot hulls
    _plot_hull(ax, orig_points, "r")
    target_hull = _plot_hull(ax, target_points, "g")
    reweighted_hull = _plot_hull(ax, reweighted_points, "b")

    if not default_scale:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    ax.set_aspect("equal")

    # Create legends for the axis
    red_patch = mpatches.Patch(color="red", label="MC")
    green_patch = mpatches.Patch(color="green", label="AmpGen")
    blue_patch = mpatches.Patch(color="blue", label="Weighted")

    ax.legend(handles=[red_patch, green_patch, blue_patch])

    ax.yaxis.set_tick_params(rotation=90)

    ax.set_xlabel(r"$Re(Z)$")
    ax.set_ylabel(r"$Im(Z)$")

    # Find distance between ampgen + reweighted hulls, and the area of the
    # reweighted hull
    def centrum(hull):
        return np.mean(hull.points, axis=0)

    def distance(hull1, hull2):
        return np.sqrt(np.sum((centrum(hull1) - centrum(hull2)) ** 2))

    return fig, ax, distance(target_hull, reweighted_hull), reweighted_hull.volume
