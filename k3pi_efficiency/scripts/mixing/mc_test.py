"""
Introduce some mixing to the MC dataframes via weighting

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
import pdg_params
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting
from lib_data import get, definitions, stats


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error

    """
    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins)
    dcs_counts, dcs_errs = stats.counts(dcs_df["time"], bins=bins, weights=dcs_wt)

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


# def _scan(
#     ratio: np.ndarray,
#     errs: np.ndarray,
#     bins: np.ndarray,
#     r_d: float,
#     params: mixing.MixingParams,
# ) -> None:
#     """
#     Do a scan
#
#     """
#     # Need x/y widths and correlations for the Gaussian constraint
#     width = 0.005
#     correlation = 0.5
#
#     n_re, n_im = 50, 51
#     n_fits = n_re * n_im
#     allowed_rez = np.linspace(-1, 1, n_re)
#     allowed_imz = np.linspace(-1, 1, n_im)
#
#     # To store the value from the fits
#     x_pulls = np.ones((n_im, n_re)) * np.inf
#     y_pulls = np.ones((n_im, n_re)) * np.inf
#     covs = np.ones((n_im, n_re, 2, 2)) * np.inf
#
#     fit_params = np.ones((n_im, n_re), dtype=object) * np.inf
#
#     chi2s = np.ones((n_im, n_re)) * np.inf
#     with tqdm(total=n_fits) as pbar:
#         for i, re_z in enumerate(allowed_rez):
#             for j, im_z in enumerate(allowed_imz):
#                 these_params = fit_util.ScanParams(
#                     r_d, params.mixing_x, params.mixing_y, re_z, im_z
#                 )
#                 scan = fitter.scan_fit(
#                     ratio, errs, bins, these_params, (width, width), correlation
#                 )
#
#                 fit_vals = scan.values
#                 fit_errs = scan.errors
#
#                 fit_params[j, i] = fit_util.ScanParams(
#                     r_d=fit_vals[0],
#                     x=fit_vals[1],
#                     y=fit_vals[2],
#                     re_z=re_z,
#                     im_z=im_z,
#                 )
#
#                 x_pulls[j, i] = (fit_vals[1] - params.mixing_x) / fit_errs[1]
#                 y_pulls[j, i] = (fit_vals[2] - params.mixing_y) / fit_errs[2]
#                 covs[j, i] = scan.covariance[1:, 1:]
#
#                 chi2s[j, i] = scan.fval
#                 pbar.update(1)
#
#     chi2s -= np.min(chi2s)
#     chi2s = np.sqrt(chi2s)
#
#     n_contours = 4
#     _plot_xy_pulls(x_pulls, y_pulls, covs, chi2s, n_contours)
#
#     fig, ax = plt.subplots(figsize=(8, 8))
#     contours = plotting.scan(
#         ax,
#         allowed_rez,
#         allowed_imz,
#         chi2s,
#         levels=np.arange(n_contours),
#     )
#
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
#     fig.colorbar(contours, cax=cbar_ax)
#     cbar_ax.set_title(r"$\sigma$")
#
#     path = "mc_mixed_scan.png"
#     print(f"saving {path}")
#     plt.savefig(path)
#     plt.close(fig)
#
#     return (allowed_rez, allowed_imz), chi2s, fit_params
#
#
# def _time_plot(
#     params: mixing.MixingParams,
#     cf_df: pd.DataFrame,
#     dcs_df: pd.DataFrame,
#     dcs_wt: np.ndarray,
#     r_d: float,
# ) -> Tuple:
#     """
#     Plot ratio of WS/RS decay times
#
#     returns loads of stuff
#
#     """
#     centres = (bins[1:] + bins[:-1]) / 2
#     widths = (bins[1:] - bins[:-1]) / 2
#
#     fig, ax = plt.subplots()
#
#     ratio, err = _ratio_err(bins, cf_df, dcs_df, None)
#     weighted_ratio, weighted_err = _ratio_err(bins, cf_df, dcs_df, dcs_wt)
#
#     ax.errorbar(centres, ratio, yerr=err, xerr=widths, label="Unweighted", fmt="k+")
#     ax.errorbar(
#         centres,
#         weighted_ratio,
#         yerr=weighted_err,
#         xerr=widths,
#         label="Weighted",
#         fmt="r+",
#     )
#
#     # No Mixing
#     best_val, _ = fitter.weighted_mean(ratio, err)
#     plotting.no_mixing(ax, best_val, "k--")
#
#     # Mixing
#     initial_guess = fit_util.MixingParams(r_d**2, 1, 1)
#     weighted_minuit = fitter.no_constraints(
#         weighted_ratio, weighted_err, bins, initial_guess
#     )
#     print(weighted_minuit)
#     # plotting.no_constraints(ax, weighted_minuit.values, "r--", "Fit (mixing)")
#
#     # Actual value
#     # We can find the expected x and y easily - they're the mixing parameters
#     expected_x = params.mixing_x
#     expected_y = params.mixing_y
#
#     # Find expected Z with a numerical integral
#     expected_z = _z(dcs_df, dcs_wt)
#
#     ideal = fit_util.ScanParams(
#         r_d=r_d,
#         x=expected_x,
#         y=expected_y,
#         re_z=expected_z.real,
#         im_z=expected_z.imag,
#     )
#     print(f"{ideal=}")
#     plotting.scan_fit(ax, ideal, "--m", "Expected Fit,\nsmall mixing approximation")
#
#     ax.set_xlabel(r"$\frac{t}{\tau}$")
#     ax.set_ylabel(r"$\frac{WS}{RS}$")
#     ax.legend()
#     ax.set_xlim(0.0, None)
#     path = "mc_mixed_times.png"
#     print(f"plotting {path}")
#     plt.savefig(path)
#     plt.close(fig)
#
#     # Do a scan as well
#     allowed_z, chi2s, fit_params = _scan(
#         weighted_ratio, weighted_err, bins, r_d, params
#     )
#
#     return (
#         weighted_ratio,
#         weighted_err,
#         centres,
#         widths,
#         allowed_z,
#         chi2s,
#         fit_params,
#         ideal,
#     )


def _mixing_weights(
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    r_d: float,
    params: mixing.MixingParams,
    q_p: Tuple[float, float],
):
    """
    Find weights to apply to the dataframe to introduce mixing

    """
    dcs_k3pi = efficiency_util.k_3pi(dcs_df)
    dcs_lifetimes = dcs_df["time"]

    # Need to find the right amount to scale the amplitudes by
    dcs_scale = r_d / np.sqrt(amplitudes.DCS_AVG_SQ)
    cf_scale = 1 / np.sqrt(amplitudes.CF_AVG_SQ)
    denom_scale = 1 / np.sqrt(amplitudes.DCS_AVG_SQ)
    mixing_weights = mixing.ws_mixing_weights(
        dcs_k3pi,
        dcs_lifetimes,
        params,
        +1,
        q_p,
        dcs_scale=dcs_scale,
        cf_scale=cf_scale,
        denom_scale=denom_scale,
    )

    # Scale weights such that their mean is right
    scale = len(dcs_df) / len(cf_df)

    return mixing_weights * scale


def _scan(
    ratio: np.ndarray, err: np.ndarray, bins: np.ndarray, ideal: fit_util.ScanParams
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Do scan of fits

    Returns arrays of chi2 and fit params and a tuple of (allowed_rez, allowed_imz)

    """
    n_re, n_im = 50, 51
    n_fits = n_re * n_im
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    # To store the value from the fits
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf
    chi2s = np.ones((n_im, n_re)) * np.inf

    # Need x/y widths and correlations for the Gaussian constraint
    width = 0.005
    correlation = 0.5

    with tqdm(total=n_fits) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    ideal.r_d, ideal.x, ideal.y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio, err, bins, these_params, (width, width), correlation
                )

                fit_vals = scan.values

                fit_params[j, i] = fit_util.ScanParams(
                    r_d=fit_vals[0],
                    x=fit_vals[1],
                    y=fit_vals[2],
                    re_z=re_z,
                    im_z=im_z,
                )

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    return chi2s, fit_params, (allowed_rez, allowed_imz)


def _scan_fits(
    ratio: np.ndarray, err: np.ndarray, bins: np.ndarray, ideal: fit_util.ScanParams
) -> None:
    """
    Plot a scan and each fit

    """
    bin_centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = (bins[1:] - bins[:-1]) / 2

    chi2s, fit_params, allowed_z = _scan(ratio, err, bins, ideal)

    # Create axes
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_xlim(bins[0], bins[-1])

    # Plot fits and scan on the axes
    contours = plotting.fits_and_scan(ax, allowed_z, chi2s, fit_params, 4)

    # Plot the errorbars now so they show up on top of the fits
    ax[0].errorbar(
        bin_centres,
        ratio,
        yerr=err,
        xerr=bin_widths,
        fmt="k+",
    )

    fig.suptitle("Weighted MC")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    # plot "ideal" fit
    plotting.scan_fit(ax[0], ideal, "--m", "Expected Fit,\nsmall mixing approximation")

    fig.savefig("mc_mixed_fits.png")


def main():
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times

    """
    # Read AmpGen dataframes
    cf_df = get.mc("2018", "cf", "magdown")
    dcs_df = get.mc("2018", "dcs", "magdown")

    # K+ only
    cf_df = cf_df[cf_df["K ID"] > 0]
    dcs_df = dcs_df[dcs_df["K ID"] > 0]

    # Time cut
    cf_keep = (0 < cf_df["time"]) & (cf_df["time"] < 7)
    dcs_keep = (0 < dcs_df["time"]) & (dcs_df["time"] < 7)
    cf_df = cf_df[cf_keep]
    dcs_df = dcs_df[dcs_keep]

    # Convert dtype
    cf_df = cf_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})
    dcs_df = dcs_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})

    bins = np.linspace(0, 7, 15)

    # Parameters determining mixing
    r_d = np.sqrt(0.003025) / 10
    print(f"{r_d**2=}")
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=pdg_params.mixing_x(),
        mixing_y=pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = _mixing_weights(cf_df, dcs_df, r_d, params, q_p)

    # Find mixed ratio and error
    ratio, err = _ratio_err(bins, cf_df, dcs_df, mixing_weights)

    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=amplitudes.AMPGEN_Z.real,
        im_z=amplitudes.AMPGEN_Z.imag,
    )

    # Make a plot showing fits and scan
    _scan_fits(ratio, err, bins, ideal)


if __name__ == "__main__":
    main()
