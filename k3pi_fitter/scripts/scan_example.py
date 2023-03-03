"""
Generate some toy data, perform fits to it fixing Z
to different values

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from tqdm import tqdm
from pulls import common

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_time_fit import util, models, fitter, plotting
from lib_data import stats


def _gen(
    domain: Tuple[float, float], params: util.ScanParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some RS and WS times

    """
    n_rs = 2000000
    gen = np.random.default_rng()

    rs_t = common.gen_rs(gen, n_rs, domain)
    ws_t = common.gen_ws(gen, n_rs, domain, models.abc_scan(params))

    return rs_t, ws_t


def _ratio_err() -> Tuple[np.ndarray, np.ndarray]:
    """
    Make some times, bin them, return their ratio and error

    """
    # Define our fit parameters
    z = (0.75, -0.25)
    params = util.ScanParams(1.0, 0.06, 0.03, *z)

    # Generate some RS and WS times
    domain = 0.0, 8.0
    rs_t, ws_t = _gen(domain, params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 20)
    rs_count, rs_err = stats.counts(values=rs_t, bins=bins)
    ws_count, ws_err = stats.counts(values=ws_t, bins=bins)

    return (*util.ratio_err(ws_count, rs_count, ws_err, rs_err), params, bins)


def _unconstrained_fit(ratio, err, params, bins):
    """
    Do an unconstrained fit

    """
    true_params = models.abc_scan(params)
    abc_fitter = fitter.no_constraints(ratio, err, bins, true_params)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    fig, axis = plt.subplots()
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.")
    plotting.no_constraints(axis, abc_fitter.values, fmt="r--", label="Fit")
    plotting.no_constraints(axis, true_params, fmt="k--", label="True")

    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"$\frac{WS}{RS}$")
    fig.suptitle("Toy data")

    axis.legend()

    fig.savefig("unconstrained_example.png")
    plt.close(fig)


def _cartesian_plot(
    ax: plt.Axes,
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    chi2s: np.ndarray,
    n_levels: int,
    true_z: Tuple[float, float],
) -> QuadContourSet:
    """
    Plot the Cartesian scan

    """
    contours = plotting.scan(
        ax,
        allowed_rez,
        allowed_imz,
        chi2s,
        levels=np.arange(n_levels),
    )
    # Plot the true/generating value of Z
    ax.plot(*true_z, "y*")

    # Plot the best-fit value of Z
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    ax.plot(allowed_rez[min_re], allowed_imz[min_im], "r*")

    # Plot a line for the best-fit points
    best_fit_im = allowed_imz[np.argmin(chi2s, axis=0)]

    def fit_fcn(x, a, b):
        """
        Straight line
        """
        return a + b * x

    fit_params, _ = curve_fit(fit_fcn, allowed_rez, best_fit_im)
    ax.plot(allowed_rez, fit_fcn(allowed_rez, *fit_params), "r--")

    ax.set_xlabel(r"Re(Z)")
    ax.set_ylabel(r"Im(Z)")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
    ax.set_title("Cartesian")

    # Legend
    ax.legend(
        handles=[
            Patch(facecolor="y", label="True"),
            Patch(facecolor="r", label="Best-Fit"),
        ]
    )

    return contours


def _true_line_plot(ax: plt.Axes, params: util.ScanParams):
    """
    Plot the expected relationship between best-fit ReZ and ImZ

    """
    points = np.linspace(*ax.get_xlim())
    expected = params.im_z + (params.y / params.x) * (params.re_z - points)
    ax.plot(points, expected, "y", linewidth=1)


def _polar_plot(
    ax: plt.Axes,
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    chi2s: np.ndarray,
    n_levels: int,
    true_z: Tuple[float, float],
):
    """
    Polar plot on an axis

    Pass args in in Cartesian co-ords, though

    """
    # Convert to polar
    xx, yy = np.meshgrid(allowed_rez, allowed_imz)
    mag = np.sqrt(xx**2 + yy**2)
    phase = np.arctan2(yy, xx)

    ax.contourf(mag, phase, chi2s, levels=np.arange(n_levels))
    ax.plot(
        [np.sqrt(true_z[0] ** 2 + true_z[1] ** 2)],
        [np.arctan2(true_z[1], true_z[0])],
        "y*",
    )

    # Plot best fit
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    ax.plot(mag[min_im, min_re], phase[min_im, min_re], "r*")

    ax.set_xlabel(r"$|Z|$")
    ax.set_ylabel(r"arg(Z)")
    ax.set_title("Polar")
    ax.plot([1, 1], ax.get_ylim(), "k-")


def main():
    """
    Generate toy data, perform fits, show plots

    """
    # ratio we'll fit to
    ratio, err, params, bins = _ratio_err()

    # Do an unconstrained fit first - just to visualise it
    _unconstrained_fit(ratio, err, params, bins)

    # Need x/y widths and correlations for the Gaussian constraint
    width = 0.005
    correlation = 0.5

    n_re, n_im = 50, 51
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio, err, bins, these_params, (width, width), correlation
                )

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    n_contours = 7
    contours = _cartesian_plot(
        ax[0], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )
    _true_line_plot(ax[0], params)

    _polar_plot(
        ax[1], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.savefig("toy_scan.png")


if __name__ == "__main__":
    main()
