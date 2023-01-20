"""
Introduce some mixing to the MC dataframes via weighting

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from fourbody.param import helicity_param
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
import pdg_params
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.plotting import phsp_labels
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting
from lib_data import get, definitions


def _z(dataframe: pd.DataFrame, weights: np.ndarray) -> complex:
    """
    Coherence factor given the desired amplitude ratio

    """
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)

    # I don't think it actually matters if we scale the amplitudes here
    cf = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, +1) / amplitudes.CF_AVG
    dcs = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, +1) / amplitudes.DCS_AVG

    # Find Z and integrals
    z = np.sum(cf * dcs.conjugate() * weights)
    num_dcs = np.sum((np.abs(dcs) ** 2) * weights)
    num_cf = np.sum((np.abs(cf) ** 2) * weights)

    return z / np.sqrt(num_dcs * num_cf)


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


def _plot_xy_pulls(
    x_pulls: np.ndarray, y_pulls: np.ndarray, covariances: np.ndarray, chi2s: np.ndarray, n_levels: int
):
    """
    Plot pulls for x and y

    """
    flat_x, flat_y, flat_chi2 = (a.ravel() for a in (x_pulls, y_pulls, chi2s))

    # TODO - do something with these
    flat_covs = covariances.reshape(len(flat_x), 2, 2)

    # Get masks of chi2<1, 1<chi2<2, etc.
    masks = (
        flat_chi2 < 1.0,
        (1.0 < flat_chi2) & (flat_chi2 < 2.0),
        (2.0 < flat_chi2) & (flat_chi2 < 3.0),
        flat_chi2 > 3.0,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_levels]
    labels = (
        r"$\chi^2 < 1.0$",
        r"$1.0 < \chi^2 < 2.0$",
        r"$2.0 < \chi^2 < 3.0$",
        r"\chi^2 > 3.0",
    )
    hist_kw = {
        "bins": np.linspace(-10, 10, 50),
        "stacked": True,
        "color": colours,
        "label": labels,
    }

    ax[0].hist([flat_x[mask] for mask in masks], **hist_kw)
    ax[1].hist([flat_y[mask] for mask in masks], **hist_kw)

    ax[0].legend()

    ax[0].set_xlabel(r"$\frac{x_{\mathrm{fit}} - x_{\mathrm{true}}}{\sigma_x}$")
    ax[1].set_xlabel(r"$\frac{y_{\mathrm{fit}} - y_{\mathrm{true}}}{\sigma_y}$")
    fig.suptitle(r"MC mixing $xy$ scan: pulls")

    fig.tight_layout()

    path = "mc_mixed_xy_scan.png"
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)


def _scan(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    r_d: float,
    params: mixing.MixingParams,
) -> None:
    """
    Do a scan

    """
    # Need x/y widths and correlations for the Gaussian constraint
    width = 0.005
    correlation = 0.5

    n_re, n_im = 50, 51
    n_fits = n_re * n_im
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    # To store the value from the fits
    x_pulls = np.ones((n_im, n_re)) * np.inf
    y_pulls = np.ones((n_im, n_re)) * np.inf
    covs = np.ones((n_im, n_re, 2, 2)) * np.inf

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_fits) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    r_d, params.mixing_x, params.mixing_y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio, errs, bins, these_params, (width, width), correlation
                )

                fit_vals = scan.values
                fit_errs = scan.errors

                x_pulls[j, i] = (fit_vals[1] - params.mixing_x) / fit_errs[1]
                y_pulls[j, i] = (fit_vals[2] - params.mixing_y) / fit_errs[2]
                covs[j, i] = scan.covariance[1:, 1:]

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    n_contours = 4
    _plot_xy_pulls(x_pulls, y_pulls, covs, chi2s, n_contours)

    fig, ax = plt.subplots(figsize=(8, 8))
    contours = plotting.scan(
        ax,
        allowed_rez,
        allowed_imz,
        chi2s,
        levels=np.arange(n_contours),
    )

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    path = "mc_mixed_scan.png"
    print(f"saving {path}")
    plt.savefig(path)
    plt.close(fig)


def _time_plot(
    params: mixing.MixingParams,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
    r_d: float,
) -> None:
    """
    Plot ratio of WS/RS decay times

    """
    bins = np.linspace(0, 7, 15)
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    fig, ax = plt.subplots()

    ratio, err = _ratio_err(bins, cf_df, dcs_df, None)
    weighted_ratio, weighted_err = _ratio_err(bins, cf_df, dcs_df, dcs_wt)

    ax.errorbar(centres, ratio, yerr=err, xerr=widths, label="Unweighted", fmt="k+")
    ax.errorbar(
        centres,
        weighted_ratio,
        yerr=weighted_err,
        xerr=widths,
        label="Weighted",
        fmt="r+",
    )

    # No Mixing
    best_val, _ = fitter.weighted_mean(ratio, err)
    plotting.no_mixing(ax, best_val, "k--")

    # Mixing
    initial_guess = fit_util.MixingParams(r_d**2, 1, 1)
    weighted_minuit = fitter.no_constraints(
        weighted_ratio, weighted_err, bins, initial_guess
    )
    print(weighted_minuit)
    # plotting.no_constraints(ax, weighted_minuit.values, "r--", "Fit (mixing)")

    # Actual value
    # We can find the expected x and y easily - they're the mixing parameters
    expected_x = params.mixing_x
    expected_y = params.mixing_y

    # Find expected Z with a numerical integral
    expected_z = _z(dcs_df, dcs_wt)

    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=expected_x,
        y=expected_y,
        re_z=expected_z.real,
        im_z=expected_z.imag,
    )
    print(f"{ideal=}")
    plotting.scan_fit(ax, ideal, "--m", "Expected Fit,\nsmall mixing approximation")

    ax.set_xlabel(r"$\frac{t}{\tau}$")
    ax.set_ylabel(r"$\frac{WS}{RS}$")
    ax.legend()
    ax.set_xlim(0.0, None)
    path = "mc_mixed_times.png"
    print(f"plotting {path}")
    plt.savefig(path)
    plt.close(fig)

    # Do a scan as well
    _scan(ratio, err, bins, r_d, params)


def _hists(cf_df: pd.DataFrame, dcs_df: pd.DataFrame, weights: np.ndarray) -> None:
    """
    save phase space histograms

    """
    cf_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(cf_df)), cf_df["time"])
    )
    dcs_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(dcs_df)), dcs_df["time"])
    )

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    hist_kw = {"histtype": "step", "density": True}
    for a, cf, dcs, label in zip(ax.ravel(), cf_pts.T, dcs_pts.T, phsp_labels()):
        contents, bins, _ = a.hist(cf, bins=100, label="CF", **hist_kw)
        a.hist(dcs, bins=bins, label="DCS", **hist_kw)
        a.hist(dcs, bins=bins, label="weighted", weights=weights, **hist_kw)
        a.set_xlabel(label)
        a.set_ylim(0, np.max(contents) * 1.1)

    ax[0, 0].legend()
    ax.ravel()[-1].legend()

    fig.savefig("mc_mixed_hists.png")
    plt.close(fig)


def _exact_dcs_integral(r_d: float, x: float, y: float, z: complex):
    """
    The integral of the exact (i.e. valid for large x, y) DCS decay rate
    assumes q = p = 1 because it's easier

    """

    def _exact_dcs(pts: np.ndarray):
        """Exact suppressed decay rate"""
        return (
            0.5
            * (
                r_d**2 * (np.cosh(y * pts) + np.cos(x * pts))
                + (np.cosh(y * pts) - np.cos(x * pts))
                + 2 * r_d * (z.real * np.sinh(y * pts) + z.imag * np.sin(x * pts))
            )
            * np.exp(-pts)
        )

    return quad(_exact_dcs, 0, np.inf)[0]


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

    # Print the mixing weight at t=0; this should be r_D^2
    weights_at_zero_time = mixing.ws_mixing_weights(
        dcs_k3pi,
        np.zeros(len(dcs_df)),
        params,
        +1,
        q_p,
        dcs_scale=dcs_scale,
        cf_scale=cf_scale,
        denom_scale=denom_scale,
    )
    print(f"{weights_at_zero_time=}")

    # _hists(cf_df, dcs_df, mixing_weights)

    # Scale weights such that their mean is right
    # This only works for small mixing
    # Want sum(wt) = N_{cf} * dcs integral / cf integral
    # cf integral = 1 since its just an exponential
    # z = _z(dcs_df, mixing_weights)
    # dcs_integral = _exact_dcs_integral(r_d, params.mixing_x, params.mixing_y, z)
    # scale = dcs_integral * len(cf_df) / (np.mean(mixing_weights) * len(mixing_weights))
    scale = len(dcs_df) / len(cf_df)

    return mixing_weights * scale


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

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

    # Parameters determining mixing
    r_d = np.sqrt(0.5)
    print(f"{r_d**2=}")
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=2 * pdg_params.mixing_x(),
        mixing_y=2 * pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = _mixing_weights(cf_df, dcs_df, r_d, params, q_p)

    _time_plot(params, cf_df, dcs_df, mixing_weights, r_d)


if __name__ == "__main__":
    main()
