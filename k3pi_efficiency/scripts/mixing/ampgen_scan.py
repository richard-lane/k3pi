"""
Introduce some mixing to the DCS AmpGen dataframe

Perform a scan of Z fits with/without this mixing introduced to see how much
difference there is

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
import pdg_params
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting, models


def _z(dataframe: pd.DataFrame, *, weights: np.ndarray = None) -> complex:
    """
    Coherence factor given the desired amplitude ratio

    """
    if weights is None:
        weights = np.ones(len(dataframe))

    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)
    cf = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, +1)
    dcs = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, +1)

    # Find Z and integrals
    z = np.sum(cf * dcs.conjugate() * weights)
    num_dcs = np.sum((np.abs(dcs) ** 2) * weights)
    num_cf = np.sum((np.abs(cf) ** 2) * weights)

    return z / np.sqrt(num_dcs * num_cf)


def _rd(*, x: float, y: float, z: complex, n_dcs: float, n_cf: float) -> float:
    """
    Expected rD from the other parameters in the equation and
    the DCS and CF statistics

    Finds rD by equating int(DCS rate) / nDCS = int(CF rate) / nCF

    Performs the integrals to infinity

    """
    # CF rate is just exponential
    cf_integral = 1

    def dcs_integral(r_d: float):
        abc_params = models.abc_scan(
            fit_util.ScanParams(r_d=r_d, x=x, y=y, re_z=z.real, im_z=z.imag)
        )
        # Assume the integral evaluated at infinity is 0... otherwise we run into trouble...?
        return -models._ws_integral_dispatcher(0, *abc_params)

    def target_fcn(r_d: float):
        return dcs_integral(r_d) - cf_integral * n_dcs / n_cf

    # Initial guess is just the sqrt of ratio of counts
    return fsolve(target_fcn, x0=np.sqrt(n_dcs / n_cf))[0]


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error

    """
    cf_counts, cf_errs = fit_util.bin_times(cf_df["time"], bins=bins)
    dcs_counts, dcs_errs = fit_util.bin_times(dcs_df["time"], bins=bins, weights=dcs_wt)
    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


def _scan_plot(
    params: mixing.MixingParams,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
) -> None:
    """
    Plot scans of AmpGen ratios with/without mixing

    """
    bins = np.concatenate((np.linspace(0, 7, 10), [11, 18]))

    n_re, n_im = 51, 50
    allowed_rez, allowed_imz = (np.linspace(-1, 1, num) for num in (n_re, n_im))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ratio, err = _ratio_err(bins, cf_df, dcs_df, None)
    weighted_ratio, weighted_err = _ratio_err(bins, cf_df, dcs_df, dcs_wt)

    # Expected values of parameters
    expected_z = _z(dcs_df, weights=dcs_wt)
    expected_rd = _rd(
        x=params.mixing_x,
        y=params.mixing_y,
        z=expected_z,
        n_dcs=np.sum(dcs_wt),
        n_cf=len(cf_df),
    )
    ideal = fit_util.ScanParams(
        r_d=expected_rd,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=expected_z.real,
        im_z=expected_z.imag,
    )

    # xy constraint width
    # Not realistic, made up
    width = 0.005
    correlation = 0.5

    # Do scans
    no_mixing_chi2 = np.ones((n_im, n_re)) * np.inf
    weighted_chi2 = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    ideal.r_d, ideal.x, ideal.y, re_z, im_z
                )
                no_mixing_fit = fitter.scan_fit(
                    ratio, err, bins, these_params, (width, width), correlation
                )
                mixing_fit = fitter.scan_fit(
                    weighted_ratio,
                    weighted_err,
                    bins,
                    these_params,
                    (width, width),
                    correlation,
                )

                no_mixing_chi2[j, i] = no_mixing_fit.fval
                weighted_chi2[j, i] = mixing_fit.fval
                pbar.update(1)

    no_mixing_chi2 -= np.min(no_mixing_chi2)
    no_mixing_chi2 = np.sqrt(no_mixing_chi2)

    weighted_chi2 -= np.min(weighted_chi2)
    weighted_chi2 = np.sqrt(weighted_chi2)

    # Plot them
    levels = np.arange(10)
    plotting.scan(ax[0], allowed_rez, allowed_imz, no_mixing_chi2, levels=levels)
    contours = plotting.scan(
        ax[1], allowed_rez, allowed_imz, weighted_chi2, levels=levels
    )

    # Expected Z
    no_mixing_z = _z(dcs_df)
    ax[0].plot(no_mixing_z.real, no_mixing_z.imag, "w*")
    ax[1].plot(expected_z.real, expected_z.imag, "w*")

    ax[0].set_title("No Mixing")
    ax[1].set_title(f"Mixing; x={params.mixing_x}, y={params.mixing_y}")

    # Colourbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.savefig("mixing_ampgen_scan.png")
    plt.show()


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

    """
    # Read AmpGen dataframes
    cf_df = efficiency_util.ampgen_df("cf", "k_plus", train=None)
    dcs_df = efficiency_util.ampgen_df("dcs", "k_plus", train=None)

    # Introduce mixing
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=pdg_params.mixing_x(),
        mixing_y=pdg_params.mixing_y(),
    )
    dcs_k3pi = efficiency_util.k_3pi(dcs_df)
    dcs_lifetimes = dcs_df["time"]
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = mixing.ws_mixing_weights(dcs_k3pi, dcs_lifetimes, params, +1, q_p)

    _scan_plot(params, cf_df, dcs_df, mixing_weights)


if __name__ == "__main__":
    main()
