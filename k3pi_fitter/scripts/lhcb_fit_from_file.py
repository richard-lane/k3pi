"""
Plot fit to WS/RS ratio from a text file of yields,
using the LHCb fitter only (i.e. no charm threshhold)

"""
import sys
import pickle
import pathlib
import argparse

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
from lib_time_fit.charm_threshhold import likelihoods


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
    fit_systematic: bool,
):
    """
    From a file of yields, time bins etc., find the yields
    and plot a fit to their ratio

    """
    if year == "all":
        assert magnetisation == "all"

    if sec_correction:
        rs_sec_frac = ipchi2_fit.sec_fracs("cf")
        ws_sec_frac = ipchi2_fit.sec_fracs("dcs")

    # Special value for phsp integrated
    if phsp_bin == -1:
        phsp_bin = None

    if year == "all":
        time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.all_yields(
            phsp_bin, bdt_cut, efficiency, alt_bkg
        )

    else:
        yield_file_path = mass_util.yield_file(
            year, magnetisation, phsp_bin, bdt_cut, efficiency, alt_bkg
        )

        assert yield_file_path.exists()

        # Get time bins, yields and errors
        time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.read_yield(
            yield_file_path
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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

    if fit_systematic:
        binned = phsp_bin is not None

        rs_errs = mass_util.systematic_pull_scale(rs_errs, "cf", binned=binned)
        ws_errs = mass_util.systematic_pull_scale(ws_errs, "dcs", binned=binned)

    ratio = ws_yields / rs_yields
    ratio_err = ratio * np.sqrt((rs_errs / rs_yields) ** 2 + (ws_errs / ws_yields) ** 2)

    # Do scan of fits
    n_re, n_im = 31, 30
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)
    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf

    charm_chi2 = np.full(chi2s.shape, np.inf)
    cleo_fcn = likelihoods.cleo_fcn()
    bes_fcn = likelihoods.bes_fcn()

    initial_rdxy = 0.0553431, definitions.CHARM_X, definitions.CHARM_Y
    xy_err = (definitions.CHARM_X_ERR, definitions.CHARM_Y_ERR)
    xy_corr = definitions.CHARM_XY_CORRELATION

    # Do a quick fit with no constraints to find rD
    no_constraints_fitter = fitter.no_constraints(ratio, ratio_err, time_bins, util.MixingParams(initial_rdxy[0], 0, 0))
    print(f"{no_constraints_fitter.valid}\t{100 * no_constraints_fitter.values[0]:.2f}\t{100 * no_constraints_fitter.errors[0]:.2f}")

    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(*initial_rdxy, re_z, im_z)

                combined_fitter = fitter.scan_fit(
                    ratio,
                    ratio_err,
                    time_bins,
                    these_params,
                    xy_err,
                    xy_corr,
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

                # Charm threshold only, if we're restricted to 1 phsp bin
                if phsp_bin is not None:
                    charm_chi2[j, i] = likelihoods.combined_chi2(
                        phsp_bin,
                        re_z,
                        im_z,
                        *initial_rdxy[1:],
                        initial_rdxy[0],
                        cleo_fcn=cleo_fcn,
                        bes_fcn=bes_fcn,
                    )

                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Plot scan, fits
    contours = plotting.fits_and_scan(
        axes, (allowed_rez, allowed_imz), chi2s, fit_params, 4
    )

    # Plot charm only scan
    charm_chi2 -= np.nanmin(charm_chi2)
    charm_chi2 = np.sqrt(charm_chi2)
    plotting.scan(
        axes[1],
        allowed_rez,
        allowed_imz,
        charm_chi2,
        [0, 1, 2, 3],
        plot_kw={"alpha": 0.5, "cmap": "Greys"},
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

    if phsp_bin is None:
        phsp_bin = "bin_integrated"

    path = f"lhcb_fits_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{phsp_bin}_{alt_bkg=}_{sec_correction=}_{misid_correction=}_{fit_systematic=}.png"
    print(f"plotting {path}")
    fig.savefig(path)
    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mass fit plots")
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018", "all"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown", "all"},
        help="magnetisation direction",
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")
    parser.add_argument(
        "--efficiency", action="store_true", help="Correct for the detector efficiency"
    )
    parser.add_argument(
        "phsp_bin",
        type=int,
        choices=[-1, *range(4)],
        help="Phase space bin index. -1 for integrated",
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
    parser.add_argument(
        "--fit_systematic",
        action="store_true",
        help="Scale the statistical errors up by a constant to account for a possible signal shape mismodelling systematic",
    )

    main(**vars(parser.parse_args()))
