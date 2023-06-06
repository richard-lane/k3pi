"""
Using the phase space integrated yields file,
plot fits using the no mixing and the mixing (unconstrained)
hypothesis

"""
import sys
import pickle
import pathlib
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as scipy_chi2

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))

from lib_data import ipchi2_fit
from lib_data.util import misid_correction as correct_misid
from libFit import util as mass_util
from lib_time_fit import plotting, util, fitter


def main(
    *,
    year: str,
    magnetisation: str,
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

    # Want the phsp-bin integrated yields
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

    # Scale the errors by the systematic pull if we need to
    if fit_systematic:
        binned = phsp_bin is not None

        rs_errs = mass_util.systematic_pull_scale(rs_errs, "cf", binned=binned)
        ws_errs = mass_util.systematic_pull_scale(ws_errs, "dcs", binned=binned)

    # Do secondary fraction correction if we need to
    if sec_correction:
        rs_yields = ipchi2_fit.correct(rs_yields, rs_sec_frac)
        rs_errs = ipchi2_fit.correct(rs_errs, rs_sec_frac)

        ws_yields = ipchi2_fit.correct(ws_yields, ws_sec_frac)
        ws_errs = ipchi2_fit.correct(ws_errs, ws_sec_frac)

    # Do misid correction if we need to
    if misid_correction:
        ws_yields, ws_errs = correct_misid(ws_yields, ws_errs)

    ratio = ws_yields / rs_yields
    ratio_err = ratio * np.sqrt((rs_errs / rs_yields) ** 2 + (ws_errs / ws_yields) ** 2)

    # First make a plot using the no mixing hypothesis, recording the chi2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for axis in axes:
        axis.set_xlim(0, time_bins[-1])

    no_mix_rd_sq, _ = fitter.weighted_mean(ratio, ratio_err)
    plotting.no_mixing(axes[0], no_mix_rd_sq, "r--")

    no_mix_chi2 = np.sum(((ratio - no_mix_rd_sq) / ratio_err) ** 2)
    axes[0].set_title(rf"No Mixing: $\chi^2=${no_mix_chi2:.3f}")

    # Then make a plot using the mixing hypothesis, recording the chi2
    unconstrained_fitter = fitter.no_constraints(
        ratio, ratio_err, time_bins, util.MixingParams(no_mix_rd_sq, 0.0, 0.0)
    )
    plotting.no_constraints(axes[1], unconstrained_fitter.values, "r--")

    mix_chi2 = unconstrained_fitter.fval
    axes[1].set_title(rf"Unconstrained Mixing: $\chi^2=${mix_chi2:.3f}")

    widths = (time_bins[1:] - time_bins[:-1]) / 2
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    for axis in axes:
        axis.errorbar(centres, ratio, xerr=widths, yerr=ratio_err, fmt="k+")
        axis.set_xlabel(r"$t/\tau$")
        axis.set_ylabel(r"$\frac{WS}{RS}$")

    # Find the difference in these chi2 values
    chi2_diff = no_mix_chi2 - mix_chi2

    # Find the p-value, given that the mixing model has 2 more degrees of freedom than the no-mixing one
    p_val = 1 - scipy_chi2.cdf(chi2_diff, 2)
    fig.suptitle(rf"$\Delta\chi^2\Rightarrow$No Mixing p value p={p_val:.3E}")

    # Indicate rD from the fit on the plot
    world_r_d = 0.05504543
    world_r_d_err = 0.000635

    tick_width = 0.0075
    r_d = unconstrained_fitter.values[0]
    r_d_err = unconstrained_fitter.errors[0]
    print(f"{r_d:.6f}+-{r_d_err:.6f}")
    axes[1].axhline(
        world_r_d**2, xmin=-tick_width, xmax=tick_width, color="r", clip_on=False
    )
    axes[1].axhline(
        (world_r_d + world_r_d_err) ** 2,
        xmin=-tick_width,
        xmax=tick_width,
        color="r",
        clip_on=False,
        alpha=0.5,
    )
    axes[1].axhline(
        (world_r_d - world_r_d_err) ** 2,
        xmin=-tick_width,
        xmax=tick_width,
        color="r",
        clip_on=False,
        alpha=0.5,
    )

    # Shade the error area
    # Fractional Error in rd^2 is twice the fractional error in rD
    # then can cancel one of the r_ds
    rd_sq_err = 2 * r_d * r_d_err
    rd_sq = r_d**2
    axes[1].fill_between(
        [0.0, 1.0],
        [rd_sq + rd_sq_err] * 2,
        [rd_sq - rd_sq_err] * 2,
        color="r",
        edgecolor=None,
        alpha=0.3,
    )
    axes[1].text(-0.5, rd_sq, r"$r_D^2$", color="r")

    fig.tight_layout()

    path = f"mixing_fits_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{alt_bkg=}_{sec_correction=}_{misid_correction=}_{fit_systematic=}.png"
    fig.savefig(path)

    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create time fit plots for the mixing/no mixing hypotheses"
    )
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
