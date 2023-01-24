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
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting
from lib_data import get, definitions, stats


def _efficiency_weights(
    dataframe: pd.DataFrame, year: str, magnetisation: str, sign: str, k_sign: str
) -> np.ndarray:
    """
    Get efficiency weights for a dataframe

    """
    # Get the right reweighter
    # TODO for now just use the DCS reweighter for both since there seems to be some issue with the relative scaling
    reweighter = get_reweighter(
        year, "dcs", magnetisation, k_sign=k_sign, fit=False, cut=False
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
    Ratio and error

    """
    # Get efficiency weights from the dataframes
    cf_eff_wt = _efficiency_weights(cf_df, year, magnetisation, "cf", k_sign)
    dcs_eff_wt = _efficiency_weights(dcs_df, year, magnetisation, "dcs", k_sign)

    # Scale efficiency weights
    cf_eff_wt, dcs_eff_wt = (
        arr[0] for arr in efficiency_util.scale_weights([cf_eff_wt], [dcs_eff_wt], 1.0)
    )

    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins, weights=cf_eff_wt)
    dcs_counts, dcs_errs = stats.counts(
        dcs_df["time"], bins=bins, weights=dcs_wt * dcs_eff_wt
    )

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


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

    fig.suptitle("Weighted MC with efficiency")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    # plot "ideal" fit
    plotting.scan_fit(ax[0], ideal, "--m", "Expected Fit,\nsmall mixing approximation")

    fig.savefig("mc_mixed_fits_eff.png")


def main():
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times with the efficiency correction to see what effect it has

    """
    # Read AmpGen dataframes
    year, magnetisation = "2018", "magdown"
    cf_df = get.mc(year, "cf", magnetisation)
    dcs_df = get.mc(year, "dcs", magnetisation)

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
    r_d = np.sqrt(0.003025)
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
    ratio, err = _ratio_err(
        bins,
        cf_df,
        dcs_df,
        mixing_weights,
        year,
        magnetisation,
        "both",  # Use the k+/k- reweighter for now
    )

    # Cut off the first bin cus its rubbish
    ratio = ratio[1:]
    err = err[1:]
    bins = bins[1:]

    # TODO find this properly
    ampgen_z = 0.6708020379588885 - 0.050344237199943055j
    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=ampgen_z.real,
        im_z=ampgen_z.imag,
    )

    # Make a plot showing fits and scan
    _scan_fits(ratio, err, bins, ideal)


if __name__ == "__main__":
    main()
