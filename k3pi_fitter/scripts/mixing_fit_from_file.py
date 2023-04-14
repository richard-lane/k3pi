"""
Using the phase space integrated yields file,
plot fits using the no mixing and the mixing (unconstrained)
hypothesis

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2 as scipy_chi2

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))

from lib_data import ipchi2_fit
from libFit import util as mass_util
from lib_time_fit import plotting, util, fitter, definitions


def main(
    *,
    year: str,
    magnetisation: str,
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
    sec_correction: bool,
):
    """
    From a file of yields, time bins etc., find the yields
    and plot a fit to their ratio

    """
    if sec_correction:
        rs_sec_frac = ipchi2_fit.sec_fracs("cf")
        ws_sec_frac = ipchi2_fit.sec_fracs("dcs")

    # Want the phsp-bin integrated yields
    yield_file_path = mass_util.yield_file(
        year, magnetisation, None, bdt_cut, efficiency, alt_bkg
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

    fig.tight_layout()

    path = f"mixing_fits_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{alt_bkg=}_{sec_correction=}.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create time fit plots for the mixing/no mixing hypotheses"
    )
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
        "--alt_bkg", action="store_true", help="Whether to use alt bkg model file"
    )
    parser.add_argument(
        "--sec_correction",
        action="store_true",
        help="Correct the yields by the secondary fractions in each time bin",
    )

    main(**vars(parser.parse_args()))
