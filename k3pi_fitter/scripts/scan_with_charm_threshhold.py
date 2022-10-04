"""
Generate some toy data, perform fits to it fixing Z
to different values

Show scans with the charm threshhold likelihood only,
the charm fit only and the combined fitter

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

from lib_time_fit import util, models, fitter, plotting
from lib_time_fit.charm_threshhold import likelihoods


def _gen(
    domain: Tuple[float, float], params: util.ScanParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some RS and WS times

    """
    n_rs = 240000000
    gen = np.random.default_rng(seed=0)

    rs_t = common.gen_rs(gen, n_rs, domain)
    ws_t = common.gen_ws(gen, n_rs, domain, models.abc_scan(params))

    return rs_t, ws_t


def _ratio_err(params: util.ScanParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make some times, bin them, return their ratio and error

    """
    # Generate some RS and WS times
    domain = 0.0, 8.0
    rs_t, ws_t = _gen(domain, params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 20)
    rs_count, rs_err = util.bin_times(rs_t, bins=bins)
    ws_count, ws_err = util.bin_times(ws_t, bins=bins)

    return (*util.ratio_err(ws_count, rs_count, ws_err, rs_err), bins)


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
    ax.plot(*true_z, "yo")

    # Plot the best-fit value of Z
    min_im, min_re = np.unravel_index(np.nanargmin(chi2s), chi2s.shape)
    ax.plot(allowed_rez[min_re], allowed_imz[min_im], "r*", alpha=0.5)

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


def _charm_only_scan(
    ax: Tuple[plt.Axes, plt.Axes],
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    bin_number: int,
    x: float,
    y: float,
    r_d: float,
    n_contours: int,
) -> Tuple[float, float]:
    """
    Scan the z plane for the chi2 values from the charm threshhold only

    Also plot them

    returns best z

    """
    n_im = len(allowed_imz)
    n_re = len(allowed_rez)

    threshhold_chi2s = np.ones((n_im, n_re)) * np.inf
    # Pre load likelihood functions from DLLs since this
    # makes evaluating the likelihood faster
    cleo_fcn = likelihoods.cleo_fcn()
    bes_fcn = likelihoods.bes_fcn()
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                threshhold_chi2s[j, i] = likelihoods.combined_chi2(
                    bin_number,
                    re_z,
                    im_z,
                    x,
                    y,
                    r_d,
                    cleo_fcn=cleo_fcn,
                    bes_fcn=bes_fcn,
                )
                pbar.update(1)

    threshhold_chi2s -= np.nanmin(threshhold_chi2s)

    # Find the best value of Z from these fits
    best_index = np.unravel_index(
        np.nanargmin(threshhold_chi2s), threshhold_chi2s.shape
    )
    best_z = (allowed_rez[best_index[1]], allowed_imz[best_index[0]])
    _cartesian_plot(
        ax[0], allowed_rez, allowed_imz, threshhold_chi2s, n_contours, best_z
    )
    _polar_plot(ax[1], allowed_rez, allowed_imz, threshhold_chi2s, n_contours, best_z)

    return best_z


def _fit_only_scan(
    ax: Tuple[plt.Axes, plt.Axes],
    ratio: np.ndarray,
    bins: np.ndarray,
    err: np.ndarray,
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    params: util.ScanParams,
    widths: Tuple[float, float],
    correlation: float,
    n_contours: int,
) -> None:
    """
    Scan the complex plane doing the fit to decay times only (i.e. no charm threshhold info)

    """
    n_im = len(allowed_imz)
    n_re = len(allowed_rez)

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio, err, bins, these_params, widths, correlation
                )
                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    _cartesian_plot(
        ax[0], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )
    _polar_plot(
        ax[1], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )


def _combined_scan(
    ax: Tuple[plt.Axes, plt.Axes],
    ratio: np.ndarray,
    bins: np.ndarray,
    err: np.ndarray,
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    params: util.ScanParams,
    widths: Tuple[float, float],
    correlation: float,
    bin_number: int,
    n_contours: int,
) -> None:
    """
    Scan the complex plane doing the fit to decay times only (i.e. no charm threshhold info)

    """
    n_im = len(allowed_imz)
    n_re = len(allowed_rez)

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.combined_fit(
                    ratio, err, bins, these_params, widths, correlation, bin_number
                )
                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    _cartesian_plot(
        ax[0], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )

    _polar_plot(
        ax[1], allowed_rez, allowed_imz, chi2s, n_contours, (params.re_z, params.im_z)
    )


def main():
    """
    Generate toy data, perform fits, show plots

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # Set some parameters
    n_re, n_im = 50, 51
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)
    bin_number = 1
    mean_x, mean_y = 0.0039183, 0.0065139
    width_x, width_y = 0.0011489, 0.00064945
    xy_correlation = -0.301
    r_d = 0.0553431
    n_contours = 7

    # Scan the charm threshhold likelihood to find the best values of z to use
    # while we do this plot it on the axis
    best_z = _charm_only_scan(
        ax[:, 0], allowed_rez, allowed_imz, bin_number, mean_x, mean_y, r_d, n_contours
    )

    # generate toy decay times for fitting to
    params = util.ScanParams(r_d, mean_x, mean_y, *best_z)
    ratio, err, bins = _ratio_err(params)

    _fit_only_scan(
        ax[:, 1],
        ratio,
        bins,
        err,
        allowed_rez,
        allowed_imz,
        params,
        (width_x, width_y),
        xy_correlation,
        n_contours,
    )

    _combined_scan(
        ax[:, 2],
        ratio,
        bins,
        err,
        allowed_rez,
        allowed_imz,
        params,
        (width_x, width_y),
        xy_correlation,
        bin_number,
        n_contours,
    )

    for a in ax[0]:
        a.set_xlim(-1, 1)
        a.set_ylim(-1, 1)

    ax[0, 0].set_title("CLEO + BES")
    ax[0, 1].set_title("LHCb (simulation)")
    ax[0, 2].set_title("LHCb (simulation), CLEO + BES")

    fig.tight_layout()
    plt.savefig("scan.png")

    plt.show()


if __name__ == "__main__":
    main()
