"""
Introduce some mixing to the MC dataframes via weighting

Also do the efficiency correction to see what difference it makes

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
import mixing_helpers
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_definitions import RS_EFF, WS_EFF
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting, models
from lib_data import get, definitions, stats


def _efficiency_weights(
    dataframe: pd.DataFrame, year: str, magnetisation: str, sign: str, k_sign: str
) -> np.ndarray:
    """
    Get efficiency weights for a dataframe

    """
    # Get the right reweighter
    reweighter = get_reweighter(
        year, sign, magnetisation, k_sign="both", fit=False, cut=False
    )

    # Find weights
    return reweighter.weights(
        efficiency_util.points(*efficiency_util.k_3pi(dataframe), dataframe["time"])
    )


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
    year: str,
    magnetisation: str,
    k_sign: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error, with efficiency correction

    """
    # Get efficiency weights from the dataframes
    cf_eff_wt = _efficiency_weights(cf_df, year, magnetisation, "cf", k_sign)
    dcs_eff_wt = _efficiency_weights(dcs_df, year, magnetisation, "dcs", k_sign)

    # Scale efficiency weights
    cf_eff_wt, dcs_eff_wt = (
        arr[0]
        for arr in efficiency_util.scale_weights(
            [cf_eff_wt], [dcs_eff_wt], WS_EFF / RS_EFF
        )
    )

    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins, weights=cf_eff_wt)
    dcs_counts, dcs_errs = stats.counts(
        dcs_df["time"], bins=bins, weights=dcs_wt * dcs_eff_wt
    )

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


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
    ax[0].set_xlim(0, bins[-1])

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

    # Plot the true value of Z
    ax[1].plot(amplitudes.AMPGEN_Z.real, amplitudes.AMPGEN_Z.imag, "y*")

    # plot "ideal" fit
    plotting.scan_fit(ax[0], ideal, "--m", "Expected Fit,\nsmall mixing approximation")
    ax[0].legend()
    ax[1].legend()

    ax[0].set_ylim(0.9 * ideal.r_d**2, 1.1 * models.scan(bins[-1], ideal))

    fig.suptitle("Weighted MC, efficiency corrected")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    fig.savefig("mc_mixed_fits_eff.png")


def main():
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times with efficiency correction

    """
    k_sign = "k_plus"

    # Read MC dataframes
    year, magnetisation = "2018", "magdown"
    cf_df = get.mc(year, "cf", magnetisation)
    dcs_df = get.mc(year, "dcs", magnetisation)

    # Time cut
    max_time = 7
    cf_keep = (0 < cf_df["time"]) & (cf_df["time"] < max_time)
    dcs_keep = (0 < dcs_df["time"]) & (dcs_df["time"] < max_time)
    cf_df = cf_df[cf_keep]
    dcs_df = dcs_df[dcs_keep]

    bins = np.linspace(0, max_time, 15)

    # Convert dtype of kinematic columns
    cf_df = cf_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})
    dcs_df = dcs_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})

    # Parameters determining mixing
    r_d = np.sqrt(0.003025)
    print(f"{r_d**2=}")
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=pdg_params.mixing_x(),
        mixing_y=pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = mixing_helpers.mixing_weights(
        cf_df, dcs_df, r_d, params, k_sign, q_p
    )

    # Find mixed ratio and error
    ratio, err = _ratio_err(
        bins,
        cf_df,
        dcs_df,
        mixing_weights,
        year,
        magnetisation,
        k_sign,
    )

    # Cut off the first bin because it's bad
    ratio = ratio[1:]
    err = err[1:]
    bins = bins[1:]

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
