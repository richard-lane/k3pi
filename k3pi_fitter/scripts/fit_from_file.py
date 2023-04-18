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
from lib_time_fit import plotting, util, fitter, definitions


def _profiles(
    allowed_z,
    chi2s,
    year,
    magnetisation,
    bdt_cut,
    efficiency,
    phsp_bin,
    alt_bkg,
    sec_correction,
    misid_correction,
) -> None:
    """
    Plot 1d profiles of chi2s

    """
    (allowed_rez, allowed_imz) = allowed_z

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    rez_chi2 = np.min(chi2s, axis=0)
    imz_chi2 = np.min(chi2s, axis=1)

    axes[0].plot(allowed_rez, rez_chi2**2)
    axes[1].plot(allowed_imz, imz_chi2**2)

    axes[0].set_ylabel(r"$\Delta \chi^2$")
    axes[0].set_xlabel(r"Re(Z)")
    axes[1].set_xlabel(r"Im(Z)")

    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(0, axes[0].get_ylim()[1])

    fig.tight_layout()

    path = f"profiles_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{phsp_bin}_{alt_bkg=}_{sec_correction=}_{misid_correction=}.png"
    fig.savefig(path)
    plt.close(fig)


def _bounding_box(
    chi2s, allowed_rez, allowed_imz, n_re, n_im
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the bounding box around the best chi2 fits, so we can focus our efforts for the fit there

    """
    # Transform to sigma
    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Find the maximum along each axis
    rez_chi2 = np.nanmin(chi2s, axis=0)
    imz_chi2 = np.nanmin(chi2s, axis=1)

    # Find where the sigma reaches a maximum
    max_sigma = 4
    allowed_rez = allowed_rez[rez_chi2 < max_sigma]
    allowed_imz = allowed_imz[imz_chi2 < max_sigma]
    rez_range = allowed_rez[:: len(allowed_rez) - 1]
    imz_range = allowed_imz[:: len(allowed_imz) - 1]

    rez_centre = (rez_range[1] - rez_range[0]) / 2
    imz_centre = (imz_range[1] - imz_range[0]) / 2

    # Make them a bit wider
    scale = 1.25
    rez_range = (rez_centre - (rez_centre - rez_range[0]) * scale), (
        rez_centre + (rez_range[1] - rez_centre) * scale
    )
    imz_range = (imz_centre - (imz_centre - imz_range[0]) * scale), (
        imz_centre + (imz_range[1] - imz_centre) * scale
    )

    return np.linspace(*rez_range, n_re), np.linspace(*imz_range, n_im)


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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    yield_file_path = mass_util.yield_file(
        year, magnetisation, phsp_bin, bdt_cut, efficiency, alt_bkg
    )

    assert yield_file_path.exists()

    # Get time bins, yields and errors
    time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.read_yield(
        yield_file_path
    )
    axes[0].set_xlim(0, time_bins[-1])

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
    allowed_rez = np.linspace(-1, 1, 11)
    allowed_imz = np.linspace(-1, 1, 11)
    chi2s = np.ones((len(allowed_rez), len(allowed_imz))) * np.inf
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

    # Do the proper scan of fits
    n_re, n_im = 21, 20

    # Linspace between first and last elements
    allowed_rez, allowed_imz = _bounding_box(
        chi2s, allowed_rez, allowed_imz, n_re, n_im
    )

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

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Plot 1d profiles
    _profiles(
        (allowed_rez, allowed_imz),
        chi2s,
        year,
        magnetisation,
        bdt_cut,
        efficiency,
        phsp_bin,
        alt_bkg,
        sec_correction,
        misid_correction,
    )

    # Plot scan, fits
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
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
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
