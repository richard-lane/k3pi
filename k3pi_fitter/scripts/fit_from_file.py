"""
Plot fit to WS/RS ratio from a text file of yields

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))

from libFit import util as mass_util
from lib_time_fit import plotting, util, fitter


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bin: int,
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
):
    """
    From a file of yields, time bins etc., find the yields
    and plot a fit to their ratio

    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    yield_file_fcn = mass_util.yield_file if not alt_bkg else mass_util.alt_yield_file
    yield_file_path = yield_file_fcn(year, magnetisation, phsp_bin, bdt_cut, efficiency)

    assert yield_file_path.exists()

    # Get time bins, yields and errors
    time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.read_yield(
        yield_file_path
    )
    axes[0].set_xlim(0, time_bins[-1])

    ratio = ws_yields / rs_yields
    ratio_err = ratio * np.sqrt((rs_errs / rs_yields) ** 2 + (ws_errs / ws_yields) ** 2)

    # Do scan of fits
    n_re, n_im = 31, 30
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)
    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf

    # TODO get these from somewhere
    initial_rdxy = 0.0055, 0.0039183, 0.0065139
    xy_err = (0.0011489, 0.00064945)
    xy_corr = -0.301
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

    # Plot scan, fits
    contours = plotting.fits_and_scan(
        axes, (allowed_rez, allowed_imz), chi2s, fit_params, 4
    )

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

    path = f"fits_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{phsp_bin}_{alt_bkg=}.png"
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

    main(**vars(parser.parse_args()))
