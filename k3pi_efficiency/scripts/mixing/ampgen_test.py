"""
Introduce some mixing to the AmpGen dataframes via weighting

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
import pdg_params
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.plotting import phsp_labels
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting


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
    bins = np.concatenate((np.linspace(0, 7, 30), np.arange(7.5, 12.5), [13, 20]))
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
    plotting.no_constraints(ax, weighted_minuit.values, "r--", "Fit (mixing)")

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
    plotting.scan_fit(ax, ideal, "--m", "True")

    ax.set_xlabel(r"$\frac{t}{\tau}$")
    ax.set_ylabel(r"$\frac{WS}{RS}$")
    ax.legend()
    plt.savefig("ampgen_mixed_times.png")

    plt.show()


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

    _, ax = plt.subplots(2, 3, figsize=(12, 8))
    hist_kw = {"histtype": "step", "density": True}
    for a, cf, dcs, label in zip(ax.ravel(), cf_pts.T, dcs_pts.T, phsp_labels()):
        contents, bins, _ = a.hist(cf, bins=100, label="CF", **hist_kw)
        a.hist(dcs, bins=bins, label="DCS", **hist_kw)
        a.hist(dcs, bins=bins, label="weighted", weights=weights, **hist_kw)
        a.set_xlabel(label)
        a.set_ylim(0, np.max(contents) * 1.1)

    ax[0, 0].legend()
    ax.ravel()[-1].legend()

    plt.savefig("ampgen_mixed_hists.png")


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
    dcs_scale = r_d * np.sqrt(amplitudes.CF_AVG_SQ / amplitudes.DCS_AVG_SQ)
    mixing_weights = mixing.ws_mixing_weights(
        dcs_k3pi, dcs_lifetimes, params, +1, q_p, cf_scale=1.0, dcs_scale=dcs_scale
    )

    _hists(cf_df, dcs_df, mixing_weights)

    # Scale weights such that their mean is right
    # Want sum(wt) = N_{cf} * dcs integral / cf integral
    # cf integral = 1 since its just an exponential
    z = _z(dcs_df, mixing_weights)
    dcs_integral = _exact_dcs_integral(r_d, params.mixing_x, params.mixing_y, z)
    scale = dcs_integral * len(cf_df) / (np.mean(mixing_weights) * len(mixing_weights))

    return mixing_weights * scale


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

    """
    # Read AmpGen dataframes
    cf_df = efficiency_util.ampgen_df("cf", "k_plus", train=None)
    dcs_df = efficiency_util.ampgen_df("dcs", "k_plus", train=None)

    # Parameters determining mixing
    r_d = np.sqrt(0.5)
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=10 * pdg_params.mixing_x(),
        mixing_y=10 * pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = _mixing_weights(cf_df, dcs_df, r_d, params, q_p)

    _time_plot(params, cf_df, dcs_df, mixing_weights, r_d)


if __name__ == "__main__":
    main()
