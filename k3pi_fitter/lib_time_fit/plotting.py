"""
Utils for plotting

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.contour import QuadContourSet

from . import models, util, parabola


def no_mixing(
    axis: plt.Axes,
    val: float,
    fmt: str = "r--",
    label: str = None,
) -> None:
    """
    Plot no mixing "fit" on an axis - i.e. a horizontal line at val

    :param axis: axis to plot on
    :param ratios:
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, val * np.ones_like(pts), fmt, label=label)


def no_constraints(
    axis: plt.Axes,
    params: util.MixingParams,
    fmt: str = "r--",
    label: str = None,
) -> None:
    """
    Plot unconstrained mixing fit on an axis.
    Uses the existing axis limits as the plotting range

    :param axis: axis to plot on
    :param params: parameters (a, b, c) as defined in models.no_constraints
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, models.no_constraints(pts, *params), fmt, label=label)


def constraints(
    axis: plt.Axes,
    params: util.ConstraintParams,
    fmt: str = "r--",
    label: str = None,
) -> None:
    """
    Plot mixing fit on an axis.
    Uses the existing axis limits as the plotting range

    :param axis: axis to plot on
    :param params: parameters from the fit
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, models.constraints(pts, params), fmt, label=label)


def scan_fit(
    axis: plt.Axes,
    params: util.ScanParams,
    fmt: str = "r--",
    label: str = None,
    plot_kw: dict = None,
) -> Tuple[plt.Figure, plt.Axes, QuadContourSet]:
    """
    Plot scan fit on an axis.
    Uses the existing axis limits as the plotting range

    :param axis: axis to plot on
    :param params: parameters from the fit
    :param fmt: format string for the plot
    :param label: label to add to legend
    :param plot_kw: additional keyword arguments for the plot

    """
    lims = axis.get_xlim()
    pts = np.linspace(*lims)

    if plot_kw is None:
        plot_kw = {}

    axis.plot(pts, models.scan(pts, params), fmt, label=label, **plot_kw)
    axis.set_xlim(lims)


def scan(
    ax: plt.Axes,
    re_z: np.ndarray,
    im_z: np.ndarray,
    chi2: np.ndarray,
    levels: np.ndarray = None,
    plot_kw: dict = None,
) -> QuadContourSet:
    """
    Plot a scan- returns the contour set with the specified levels (default levels otherwise)

    """
    if plot_kw is None:
        plot_kw = {}

    return ax.contourf(*np.meshgrid(re_z, im_z), chi2, levels=levels, **plot_kw)


def _plot_fits(
    axis: plt.Axes,
    fit_vals: np.ndarray,
    chi2s: np.ndarray,
    colours: np.ndarray,
) -> None:
    """
    Plot a scan of fits on an axis, colour-coded according to the chi2 of each

    """
    # TODO make it work with variable n_levels

    for params, chi2 in zip(fit_vals.ravel(), chi2s.ravel()):
        if chi2 < 1.0:
            colour = colours[0]
            alpha = 0.3
        elif chi2 < 2.0:
            colour = colours[1]
            alpha = 0.2
        elif chi2 < 3.0:
            colour = colours[2]
            alpha = 0.05

        if chi2 < 3.0:
            scan_fit(
                axis,
                params,
                fmt="--",
                label=None,
                plot_kw={"alpha": alpha, "color": colour},
            )


def fits_and_scan(
    axes: Tuple[plt.Axes, plt.Axes],
    allowed_z: Tuple[np.ndarray, np.ndarray],
    chi2s: np.ndarray,
    fit_vals: np.ndarray,
    n_contours: int,
) -> QuadContourSet:
    """
    Plot fits and a scan on a plot

    :param axes: tuple of axes for (fit plot, scan plot)
    :param allowed_z: tuple of allowed values for (re_z, im_z)
    :param chi2s: 2d array of chi2s for the fits
    :param fit_vals: 2d array of fitter values from the fits
    :param n_coutours: 1 more than the number of contours lol

    :returns: contours

    """
    fit_ax, scan_ax = axes
    allowed_rez, allowed_imz = allowed_z

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_contours]

    _plot_fits(fit_ax, fit_vals, chi2s, colours)

    contours = scan(
        scan_ax,
        allowed_rez,
        allowed_imz,
        chi2s,
        levels=np.arange(n_contours),
        plot_kw={"colors": colours},
    )

    scan_ax.set_xlabel("Re(Z)")
    scan_ax.set_ylabel("Im(Z)")

    fit_ax.set_xlabel(r"t/$\tau$")
    fit_ax.set_ylabel(r"$\frac{WS}{RS}$")

    # Plot the best fit value
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    scan_ax.plot(allowed_rez[min_re], allowed_imz[min_im], "r*", label="Best Fit")

    # Draw a circle
    scan_ax.add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))

    return contours


def projections(
    allowed_z: Tuple[np.ndarray, np.ndarray], chi2s: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 1d profile chi2 projections of a scan

    :param allowed_z: tuple of allowed rez, allowed imZ points
    :param chi2s: 2d array of chi2s (not sqrted)

    """
    allowed_rez, allowed_imz = allowed_z

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

    min_chi2 = np.min(chi2s)
    rez_chi2 = np.min(chi2s, axis=0) - min_chi2
    imz_chi2 = np.min(chi2s, axis=1) - min_chi2

    axes[0].plot(allowed_rez, rez_chi2)
    axes[1].plot(allowed_imz, imz_chi2)

    axes[0].set_ylabel(r"$\Delta \chi^2$")
    axes[0].set_xlabel(r"Re(Z)")
    axes[1].set_xlabel(r"Im(Z)")

    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(0, axes[0].get_ylim()[1])

    fig.tight_layout()

    return fig, axes


def surface(
    allowed_z: Tuple[np.ndarray, np.ndarray], chi2s: np.ndarray, fit_vals: np.ndarray
):
    """
    Plot 2d surface plot of chi2 projections and show it

    :param allowed_z: tuple of allowed rez, allowed imZ points
    :param chi2s: 2d array of chi2s (not sqrted)
    :param fit_vals: parabola fit vals

    """
    allowed_rez, allowed_imz = allowed_z

    fig, axis = plt.subplots(subplot_kw={"projection": "3d"})

    delta_chi2s = chi2s - np.min(chi2s)

    max_chi2 = 49
    # Make the plot nicer by assigning vals that are too big to NaN
    delta_chi2s[delta_chi2s > max_chi2] = np.nan

    pts = np.meshgrid(*allowed_z)
    axis.plot_surface(*pts, delta_chi2s, linewidth=0, cmap=cm.coolwarm)

    # Plot also the fit vals
    vals = parabola.parabola(fit_vals, *allowed_z)
    axis.plot_surface(*pts, vals, linewidth=0.1, cmap=cm.gray, alpha=0.5)

    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(0, max_chi2)

    axis.set_xlabel("Re(Z)")
    axis.set_ylabel("Im(Z)")
    axis.set_zlabel(r"$\Delta \chi^2$")

    plt.show()
