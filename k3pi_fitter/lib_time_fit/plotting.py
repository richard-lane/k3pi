"""
Utils for plotting

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet

from . import models, util


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
) -> Tuple[plt.Figure, plt.Axes, QuadContourSet]:
    """
    Plot scan fit on an axis.
    Uses the existing axis limits as the plotting range

    :param axis: axis to plot on
    :param params: parameters from the fit
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, models.scan(pts, params), fmt, label=label)


def scan(
    ax: plt.Axes,
    re_z: np.ndarray,
    im_z: np.ndarray,
    chi2: np.ndarray,
    levels=None,
) -> QuadContourSet:
    """
    Plot a scan- returns the contour set with the specified levels (default levels otherwise)

    """
    return ax.contourf(*np.meshgrid(re_z, im_z), chi2, levels=levels)
