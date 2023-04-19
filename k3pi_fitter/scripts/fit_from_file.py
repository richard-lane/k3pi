"""
Plot fit to WS/RS ratio from a text file of yields

"""
import sys
import pathlib
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))

from lib_data import ipchi2_fit
from lib_data.util import misid_correction as correct_misid
from libFit import util as mass_util
from lib_time_fit import plotting, util, fitter, definitions, parabola


def _bounding_box(
    chi2s, allowed_rez, allowed_imz, n_re, n_im
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the bounding box around the best chi2 fits, so we can focus our efforts for the fit there

    """
    # Transform to sigma
    sigma = chi2s - np.nanmin(chi2s)
    sigma = np.sqrt(sigma)

    # Find the maximum along each axis
    rez_sigma = np.nanmin(sigma, axis=0)
    imz_sigma = np.nanmin(sigma, axis=1)

    # Find where the sigma reaches a maximum
    max_sigma = 4
    allowed_rez = allowed_rez[rez_sigma < max_sigma]
    allowed_imz = allowed_imz[imz_sigma < max_sigma]
    rez_range = allowed_rez[:: len(allowed_rez) - 1]
    imz_range = allowed_imz[:: len(allowed_imz) - 1]

    rez_centre = (rez_range[1] + rez_range[0]) / 2
    imz_centre = (imz_range[1] + imz_range[0]) / 2

    # Make them a bit wider
    scale = 1.3
    rez_range = (rez_centre - (rez_centre - rez_range[0]) * scale), (
        rez_centre + (rez_range[1] - rez_centre) * scale
    )
    imz_range = (imz_centre - (imz_centre - imz_range[0]) * scale), (
        imz_centre + (imz_range[1] - imz_centre) * scale
    )

    # Dont go above 1 or below -1
    rez_range = (
        rez_range[0] if rez_range[0] > -1 else -1,
        rez_range[1] if rez_range[1] < 1 else 1,
    )
    imz_range = (
        imz_range[0] if imz_range[0] > -1 else -1,
        imz_range[1] if imz_range[1] < 1 else 1,
    )

    return np.linspace(*rez_range, n_re), np.linspace(*imz_range, n_im)


def _initial_scan(
    initial_rdxy: Tuple,
    xy_err_corr: Tuple,
    ratio: np.ndarray,
    ratio_err: np.ndarray,
    time_bins: np.ndarray,
    n_re_im: Tuple,
    phsp_bin: int,
) -> Tuple[Tuple, Tuple]:
    """
    Do an initial scan of fits to find out which region we should focus our efforts on

    Returns two ranges of ReZ, ImZ values to focus our efforts on

    :param initial_rdxy: initial guess at (rd, x, y)
    :param xy_err_corr: error and correlation on x and y
    :param ratio: WS/RS ratio in each bin
    :param ratio_err: error on the WS/RS ratio in each bin
    :param time_bins: time bins
    :param n_re_im: number of re and im z we want in our grid
    :param phsp_bin: bin number

    :returns: array of allowed rez points
    :returns: array of allowed imz points

    """
    xy_err, xy_corr = xy_err_corr
    n_re, n_im = n_re_im

    allowed_rez = np.linspace(-1, 1, 20)
    allowed_imz = np.linspace(-1, 1, 21)
    chi2s = np.ones((len(allowed_imz), len(allowed_rez))) * np.inf
    with tqdm(total=len(allowed_imz) * len(allowed_rez)) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(*initial_rdxy, re_z, im_z)

                combined_fitter = fitter.combined_fit(
                    ratio,
                    ratio_err,
                    time_bins,
                    these_params,
                    xy_err,
                    xy_corr,
                    phsp_bin,
                )
                chi2s[j, i] = combined_fitter.fval

                pbar.update(1)

    # Use these chi2s to find the region of interest
    return _bounding_box(chi2s, allowed_rez, allowed_imz, n_re, n_im)


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bin: int,
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
    sec_correction: bool,
    misid_correction: bool,
):
    """
    From a file of yields, time bins etc., find the yields
    and plot a fit to their ratio

    """
    if sec_correction:
        rs_sec_frac = ipchi2_fit.sec_fracs("cf")
        ws_sec_frac = ipchi2_fit.sec_fracs("dcs")

    yield_file_path = mass_util.yield_file(
        year, magnetisation, phsp_bin, bdt_cut, efficiency, alt_bkg
    )

    assert yield_file_path.exists()

    # Get time bins, yields and errors
    time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.read_yield(
        yield_file_path
    )

    # Do secondary fraction correction if we need to
    if sec_correction:
        rs_yields = ipchi2_fit.correct(rs_yields, rs_sec_frac)
        rs_errs = ipchi2_fit.correct(rs_errs, rs_sec_frac)

        ws_yields = ipchi2_fit.correct(ws_yields, ws_sec_frac)
        ws_errs = ipchi2_fit.correct(ws_errs, ws_sec_frac)

    # Do double misID correction if we need to
    if misid_correction:
        ws_yields, ws_errs = correct_misid(ws_yields, ws_errs)

    ratio = ws_yields / rs_yields
    ratio_err = ratio * np.sqrt((rs_errs / rs_yields) ** 2 + (ws_errs / ws_yields) ** 2)

    # Do a quick scan of fits to determine where we should focus our efforts
    initial_rdxy = 0.0055, definitions.CHARM_X, definitions.CHARM_Y
    xy_err = (definitions.CHARM_X_ERR, definitions.CHARM_Y_ERR)
    xy_corr = definitions.CHARM_XY_CORRELATION

    n_re, n_im = 30, 31  # The number of points we want for the actual scan
    allowed_rez, allowed_imz = _initial_scan(
        initial_rdxy,
        (xy_err, xy_corr),
        ratio,
        ratio_err,
        time_bins,
        (n_re, n_im),
        phsp_bin,
    )

    # Do the proper scan of fits
    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(*initial_rdxy, re_z, im_z)

                combined_fitter = fitter.combined_fit(
                    ratio,
                    ratio_err,
                    time_bins,
                    these_params,
                    xy_err,
                    xy_corr,
                    phsp_bin,
                )
                chi2s[j, i] = combined_fitter.fval
                fit_vals = combined_fitter.values
                fit_params[j, i] = util.ScanParams(
                    r_d=fit_vals[0],
                    x=fit_vals[1],
                    y=fit_vals[2],
                    re_z=re_z,
                    im_z=im_z,
                )

                pbar.update(1)

    # Find the best fit value of Z
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    best_z = allowed_rez[min_re], allowed_imz[min_im]
    print(f"{best_z=}")

    # Fit a 2d parabola to the chi2 to get width, error, correlation out
    max_chi2 = 9
    params, errs = parabola.fit(chi2s, allowed_rez, allowed_imz, best_z, max_chi2)
    for param, err, label in zip(
        params, errs, ["ReZ", "ImZ", "ReZ width", "ImZ width", "corr"]
    ):
        print(f"{label}\t= {param:.3f} +- {err:.3f}")

    # Plot 1d profiles
    fig, axes = plotting.projections((allowed_rez, allowed_imz), chi2s)
    parabola.plot_projection(axes, params, max_chi2)
    path = f"profiles_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{phsp_bin}_{alt_bkg=}_{sec_correction=}_{misid_correction=}.png"
    axes[0].legend()
    print(f"plotting {path}")
    fig.savefig(path)
    plt.close(fig)

    # Plot a 2d landscape
    # plotting.surface((allowed_rez, allowed_imz), chi2s, params)

    # Convert chi2s to sigma for doing 2d plot
    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Plot scan, fits
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_xlim(0, time_bins[-1])
    contours = plotting.fits_and_scan(
        axes, (allowed_rez, allowed_imz), chi2s, fit_params, 4
    )
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)

    # Plot ratio on the axis
    widths = (time_bins[1:] - time_bins[:-1]) / 2
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    axes[0].errorbar(centres, ratio, xerr=widths, yerr=ratio_err, fmt="k+")
    fig.tight_layout()

    # Plot colourbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    path = f"fits_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{phsp_bin}_{alt_bkg=}_{sec_correction=}_{misid_correction=}.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mass fit plots")
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")
    parser.add_argument(
        "--efficiency", action="store_true", help="Correct for the detector efficiency"
    )
    parser.add_argument(
        "phsp_bin", type=int, choices=range(4), help="Phase space bin index"
    )
    parser.add_argument(
        "--alt_bkg", action="store_true", help="Whether to use alt bkg model file"
    )
    parser.add_argument(
        "--sec_correction",
        action="store_true",
        help="Correct the yields by the secondary fractions in each time bin",
    )
    parser.add_argument(
        "--misid_correction",
        action="store_true",
        help="Correct the yields by the double misID fraction",
    )

    main(**vars(parser.parse_args()))
