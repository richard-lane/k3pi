"""
Helper fcns for the mixing scripts, because lots of code is being
repeated

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
from lib_efficiency import mixing, efficiency_util
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import models, fitter, plotting
from lib_time_fit import util as fit_util


def mixing_weights(
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    r_d: float,
    params: mixing.MixingParams,
    k_sign: int,
    q_p: Tuple[float, float],
):
    """
    Find weights to apply to the dataframe to introduce mixing

    """
    assert k_sign in {"both", "k_plus", "k_minus"}
    print(f"\tfinding mixing weights: {len(dcs_df)=}")

    dcs_lifetimes = dcs_df["time"]

    # Need to find the right amount to scale the amplitudes by
    dcs_scale = r_d / np.sqrt(amplitudes.DCS_AVG_SQ)
    cf_scale = 1 / np.sqrt(amplitudes.CF_AVG_SQ)
    denom_scale = 1 / np.sqrt(amplitudes.DCS_AVG_SQ)

    # Simple case where we only have K+ or K- in our dataframe(s)
    if not k_sign == "both":
        dcs_k3pi = efficiency_util.k_3pi(dcs_df)
        weights = mixing.ws_mixing_weights(
            dcs_k3pi,
            dcs_lifetimes,
            params,
            +1 if k_sign == "k_plus" else -1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

    # More complicated case where we could have both
    else:
        weights = np.ones(len(dcs_df)) * np.inf
        plus_mask = dcs_df["K ID"] == 321
        print(f"\tK+ {np.sum(plus_mask)=}")
        print(f"\tK- {np.sum(~plus_mask)=}")

        # K+ weights
        weights[plus_mask] = mixing.ws_mixing_weights(
            efficiency_util.k_3pi(dcs_df[plus_mask]),
            dcs_lifetimes[plus_mask],
            params,
            +1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

        # K- weights
        weights[~plus_mask] = mixing.ws_mixing_weights(
            efficiency_util.k_3pi(dcs_df[~plus_mask]),
            dcs_lifetimes[~plus_mask],
            params,
            -1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

    # Scale weights such that their mean is right
    scale = len(cf_df) / len(dcs_df)

    return weights * scale


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


def scan_fits(
    fig: plt.Figure,
    axes: Tuple[plt.Axes, plt.Axes],
    ratio: np.ndarray,
    err: np.ndarray,
    bins: np.ndarray,
    ideal: fit_util.ScanParams,
    path: str,
) -> None:
    """
    Plot a scan and each fit

    """
    bin_centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = (bins[1:] - bins[:-1]) / 2

    chi2s, fit_params, allowed_z = _scan(ratio, err, bins, ideal)

    # Create axes
    axes[0].set_xlim(0, bins[-1])

    # Plot fits and scan on the axes
    contours = plotting.fits_and_scan(axes, allowed_z, chi2s, fit_params, 4)

    # Plot the errorbars now so they show up on top of the fits
    axes[0].errorbar(
        bin_centres,
        ratio,
        yerr=err,
        xerr=bin_widths,
        fmt="k+",
    )

    # Plot the true value of Z
    axes[1].plot(amplitudes.AMPGEN_Z.real, amplitudes.AMPGEN_Z.imag, "y*")

    # plot "ideal" fit
    plotting.scan_fit(
        axes[0], ideal, "--m", "Expected Fit,\nsmall mixing approximation"
    )
    axes[0].legend()
    axes[1].legend()

    axes[0].set_ylim(0.9 * ideal.r_d**2, 1.1 * models.scan(bins[-1], ideal))

    fig.suptitle(path)
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    print(f"saving {path}")
    fig.savefig(path)
