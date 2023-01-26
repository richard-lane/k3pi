"""
Introduce some mixing to the DCS AmpGen dataframe via weighting,
get rid of some events according to an efficiency model,
train a reweighter to correct for the efficiency (without mixing!),
perform a scan indicating the "true" value of the mixing parameter
compared to the value we get when we don't do the efficiency correction

"""
import sys
import pickle
import pathlib
from typing import Tuple
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(
    str(
        pathlib.Path(__file__).resolve().parents[2]
        / "k3pi_efficiency"
        / "scripts"
        / "mixing"
    )
)
import pdg_params
from lib_efficiency import efficiency_util, mixing, efficiency_definitions
from lib_efficiency.plotting import phsp_labels
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_efficiency.time_fitter import pdf as time_pdf
from lib_time_fit import fitter, plotting
from lib_time_fit import util as fit_util


PdfParams = namedtuple("PdfParams", ["t0", "n", "m", "a", "b", "k"])


def _z(dataframe: pd.DataFrame, weights: np.ndarray) -> complex:
    """
    Coherence factor

    """
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)

    cf_amp = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, +1)
    dcs_amp = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, +1)

    # Find Z and integrals
    cross_term = np.sum(cf_amp * dcs_amp.conjugate() * weights)
    num_dcs = np.sum((np.abs(dcs_amp) ** 2) * weights)
    num_cf = np.sum((np.abs(cf_amp) ** 2) * weights)

    return cross_term / np.sqrt(num_dcs * num_cf)


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


def _exact_dcs_integral(
    r_d: float, mixing_x: float, mixing_y: float, interference_z: complex
):
    """
    The integral of the exact (i.e. valid for large x, y) DCS decay rate

    assumes q = p = 1 because it's easier

    """

    def _exact_dcs(pts: np.ndarray):
        """Exact suppressed decay rate"""
        return (
            0.5
            * (
                r_d**2 * (np.cosh(mixing_y * pts) + np.cos(mixing_x * pts))
                + (np.cosh(mixing_y * pts) - np.cos(mixing_x * pts))
                + 2
                * r_d
                * (
                    interference_z.real * np.sinh(mixing_y * pts)
                    + interference_z.imag * np.sin(mixing_x * pts)
                )
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
    dcs_scale = r_d * np.sqrt(amplitudes.DCS_AVG_SQ / amplitudes.CF_AVG_SQ)
    mixing_weights = mixing.ws_mixing_weights(
        dcs_k3pi, dcs_lifetimes, params, +1, q_p, cf_scale=1.0, dcs_scale=dcs_scale
    )

    # Scale weights such that their mean is right
    # Want sum(wt) = N_{cf} * dcs integral / cf integral
    # cf integral = 1 since its just an exponential
    interference_z = _z(dcs_df, mixing_weights)
    dcs_integral = _exact_dcs_integral(
        r_d, params.mixing_x, params.mixing_y, interference_z
    )
    scale = dcs_integral * len(cf_df) / (np.mean(mixing_weights) * len(mixing_weights))

    return mixing_weights * scale


def _realistic_efficiency_mask(
    rng: np.random.Generator, dataframe: pd.DataFrame
) -> np.ndarray:
    """
    Boolean mask of events to keep following efficiency reweighting

    May be unused if I'm using the toy efficiency instead

    """
    # Open an efficiency reweighter
    # Doesn't really matter which one
    # Doesn't do BDT cut because AmpGen doesn't have BDT training var information
    reweighter_path = efficiency_definitions.reweighter_path(
        "2018",
        "dcs",
        "magdown",
        "both",
        time_fit=False,
        cut=False,
    )
    print(f"Opening reweighter at {reweighter_path}")
    with open(reweighter_path, "rb") as weighter_f:
        reweighter = pickle.load(weighter_f)

    # Find predicted weights for this dataframe
    # Zero means we definitely throw it away
    pred_wt = reweighter.weights(
        np.column_stack(
            (helicity_param(*efficiency_util.k_3pi(dataframe)), dataframe["time"])
        )
    )

    # Init array of False
    n_evts = len(dataframe)
    retval = np.zeros(n_evts, dtype=np.bool_)

    # This cutoff will give us a slightly different efficiency to what the model
    # describes but that's ok
    cutoff = 7.5

    # 1 / weight is proportional to how likely we are to keep it
    # Throw away all 0 weights
    nonzero_wt = pred_wt != 0
    retval[nonzero_wt] = (
        cutoff * rng.random(np.sum(nonzero_wt)) < 1 / pred_wt[nonzero_wt]
    )
    return retval


def _toy_efficiency_mask(
    rng: np.random.Generator, dataframe: pd.DataFrame, params: PdfParams
) -> np.ndarray:
    """
    Boolean mask of events to keep following efficiency reweighting

    May be unused if I'm using the realistic efficiency instead

    """
    # Evaluate time efficiency model fcn at each decay time
    pdf_vals = time_pdf(dataframe["time"], *params) * np.exp(dataframe["time"])

    # Random numbers between 0 and the max of this function
    rand = np.max(pdf_vals) * rng.random(len(dataframe))

    # Keep if random < fcn
    return rand < pdf_vals


def _reweighter(target_df: pd.DataFrame, orig_df: pd.DataFrame) -> EfficiencyWeighter:
    """
    Train + return an efficiency reweighter

    """
    target_pts = np.column_stack(
        (
            helicity_param(*efficiency_util.k_3pi(target_df[target_df["train"]])),
            target_df[target_df["train"]]["time"],
        )
    )
    orig_pts = np.column_stack(
        (
            helicity_param(*efficiency_util.k_3pi(orig_df[orig_df["train"]])),
            orig_df[orig_df["train"]]["time"],
        )
    )

    train_kwargs = {
        "n_estimators": 1,
        "max_depth": 5,
        "learning_rate": 0.7,
        "min_samples_leaf": 1800,
        "n_bins": 20000,
        "n_neighs": 10.0,
    }
    return EfficiencyWeighter(
        target_pts,
        orig_pts,
        fit=False,
        min_t=efficiency_definitions.MIN_TIME,
        **train_kwargs,
    )


def _projections(
    target_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    efficiency_wt: np.ndarray,
    path: str,
    mixing_wt: np.ndarray = None,
) -> None:
    """
    Plot projections of testing data after the reweighting with/without mixing added

    Only plots points above a minimum time

    """
    target_test = target_df[~target_df["train"]]
    orig_test = orig_df[~orig_df["train"]]

    target_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(target_test)), target_test["time"])
    )
    orig_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(orig_test)), orig_test["time"])
    )
    hist_wt = (
        efficiency_wt[~orig_df["train"]]
        if mixing_wt is None
        else efficiency_wt[~orig_df["train"]] * mixing_wt[~orig_df["train"]]
    )

    hist_wt = hist_wt[orig_pts[:, -1] > efficiency_definitions.MIN_TIME]
    target_pts = target_pts[target_pts[:, -1] > efficiency_definitions.MIN_TIME]
    orig_pts = orig_pts[orig_pts[:, -1] > efficiency_definitions.MIN_TIME]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    hist_kw = {"histtype": "step", "density": True}
    for axis, target, orig, label in zip(
        axes.ravel(), target_pts.T, orig_pts.T, phsp_labels()
    ):
        contents, bins, _ = axis.hist(target, bins=100, label="Target", **hist_kw)
        axis.hist(orig, bins=bins, label="Original", **hist_kw)
        axis.hist(orig, bins=bins, label="Weighted", weights=hist_wt, **hist_kw)

        axis.set_xlabel(label)
        axis.set_ylim(0, np.max(contents) * 1.1)

    axes[0, 0].legend()
    axes.ravel()[-1].legend()

    plt.savefig(path)
    plt.close(fig)


def _ratio_err(
    bins: np.ndarray,
    cf_t: pd.DataFrame,
    dcs_t: pd.DataFrame,
    cf_wt: np.ndarray = None,
    dcs_wt: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error

    """
    cf_counts, cf_errs = fit_util.bin_times(cf_t, bins=bins, weights=cf_wt)
    dcs_counts, dcs_errs = fit_util.bin_times(dcs_t, bins=bins, weights=dcs_wt)

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


def _plot_on_axis(
    axis: plt.Axes,
    bins: np.ndarray,
    label: str,
    fmt: str,
    dcs_t: np.ndarray,
    cf_t: np.ndarray,
    dcs_wt: np.ndarray,
    cf_wt: np.ndarray,
):
    """
    Plot the ratio on an axis

    """
    ratio, err = _ratio_err(bins, cf_t, dcs_t, cf_wt, dcs_wt)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt=fmt)

    # Plot fits
    axis.set_xlim(0, bins[-1])
    fit = fitter.no_constraints(ratio, err, bins, fit_util.MixingParams(ratio[0], 1, 1))
    plotting.no_constraints(axis, fit.values, fmt.replace("+", "-"), label)


def _plot_ratios(
    dcs_t: np.ndarray,
    cf_t: np.ndarray,
    dcs_efficiency_mask: np.ndarray,
    cf_efficiency_mask: np.ndarray,
    dcs_mixing_wt: np.ndarray,
    dcs_efficiency_wt: np.ndarray,
    cf_efficiency_wt: np.ndarray,
    params: Tuple[float, complex, mixing.MixingParams],
):
    """
    Plot ratios of DCS/CF times before/after mixing

    """
    bins = np.concatenate(
        ([0], np.linspace(efficiency_definitions.MIN_TIME, 8, 8), [18.0])
    )
    _, axis = plt.subplots()

    # Plot the ratio of times before anything
    # _plot_on_axis(axis, bins, "No Mixing", "k+", dcs_t, cf_t, None, None)

    # Plot the ratio of times after mixing
    _plot_on_axis(axis, bins, "Mixing", "b+", dcs_t, cf_t, dcs_mixing_wt, None)

    # Plot the ratio of times after efficiency applied
    _plot_on_axis(
        axis,
        bins[1:],
        "Efficiency Applied",
        "r+",
        dcs_t[dcs_efficiency_mask],
        cf_t[cf_efficiency_mask],
        dcs_mixing_wt[dcs_efficiency_mask],
        None,
    )

    # Plot the ratio of times after efficiency corrected for
    _plot_on_axis(
        axis,
        bins[1:],
        "Efficiency Corrected",
        "g+",
        dcs_t[dcs_efficiency_mask],
        cf_t[cf_efficiency_mask],
        dcs_mixing_wt[dcs_efficiency_mask] * dcs_efficiency_wt,
        cf_efficiency_wt,
    )

    # Plot the ideal ratio
    ideal = fit_util.ScanParams(
        r_d=params[0],
        x=params[2].mixing_x,
        y=params[2].mixing_y,
        re_z=params[1].real,
        im_z=params[1].imag,
    )
    print(f"{ideal=}")
    plotting.scan_fit(axis, ideal, "--b", "True")

    axis.legend()
    axis.set_ylabel(r"t/ $\tau$")
    axis.set_xlabel(r"$\frac{WS}{RS}$")
    plt.savefig("toy_ratios.png")
    plt.show()


def _scan(
    dcs_t: np.ndarray,
    cf_t: np.ndarray,
    dcs_efficiency_mask: np.ndarray,
    cf_efficiency_mask: np.ndarray,
    dcs_mixing_wt: np.ndarray,
    dcs_eff_wt: np.ndarray,
    cf_eff_wt: np.ndarray,
    params: Tuple[float, complex, mixing.MixingParams],
) -> None:
    """
    Plot a scan of the ratio between DCS and CF dataframes

    """
    # Measure ratios
    bins = np.concatenate(
        ([0], np.linspace(efficiency_definitions.MIN_TIME, 8, 8), [18.0])
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Before Efficiency
    ratio_before, err_before = _ratio_err(bins, cf_t, dcs_t, None, dcs_mixing_wt)

    # After Efficiency
    ratio_after, err_after = _ratio_err(
        bins[1:],
        cf_t[cf_efficiency_mask],
        dcs_t[dcs_efficiency_mask],
        None,
        dcs_mixing_wt[dcs_efficiency_mask],
    )

    # Corrected for Efficiency
    ratio_corrected, err_corrected = _ratio_err(
        bins[1:],
        cf_t[cf_efficiency_mask],
        dcs_t[dcs_efficiency_mask],
        cf_eff_wt,
        dcs_eff_wt * dcs_mixing_wt[dcs_efficiency_mask],
    )

    # xy constraint width
    # Not realistic, made up
    width = 0.005
    correlation = 0.5

    # Iterate over values
    n_re, n_im = 100, 101
    allowed_rez, allowed_imz = (np.linspace(-1, 1, num) for num in (n_re, n_im))
    before_chi2 = np.ones((n_im, n_re)) * np.inf
    after_chi2 = np.ones((n_im, n_re)) * np.inf
    corrected_chi2 = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    params[0],
                    params[2].mixing_x,
                    params[2].mixing_y,
                    re_z,
                    im_z,
                )
                before_fit = fitter.scan_fit(
                    ratio_before,
                    err_before,
                    bins,
                    these_params,
                    (width, width),
                    correlation,
                )
                after_fit = fitter.scan_fit(
                    ratio_after,
                    err_after,
                    bins[1:],
                    these_params,
                    (width, width),
                    correlation,
                )
                corrected_fit = fitter.scan_fit(
                    ratio_corrected,
                    err_corrected,
                    bins[1:],
                    these_params,
                    (width, width),
                    correlation,
                )

                before_chi2[j, i] = before_fit.fval
                after_chi2[j, i] = after_fit.fval
                corrected_chi2[j, i] = corrected_fit.fval
                pbar.update(1)

    # Normalise + plot chi2s
    before_chi2 -= np.min(before_chi2)
    after_chi2 -= np.min(after_chi2)
    corrected_chi2 -= np.min(corrected_chi2)

    levels = np.arange(10)
    plotting.scan(axes[0], allowed_rez, allowed_imz, before_chi2, levels=levels)
    plotting.scan(axes[1], allowed_rez, allowed_imz, after_chi2, levels=levels)
    contours = plotting.scan(
        axes[2], allowed_rez, allowed_imz, corrected_chi2, levels=levels
    )

    # Plot Z
    for a in axes:
        a.plot(params[1].real, params[1].imag, "y*")

    axes[0].set_title("No Efficiency")
    axes[1].set_title("Efficiency Applied")
    axes[2].set_title("Efficiency Corrected")

    # Colourbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.savefig("toy_scans.png")
    plt.show()


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

    """
    # Read AmpGen dataframes
    num = None
    cf_df = efficiency_util.ampgen_df("cf", "k_plus", train=None)[:num]
    dcs_df = efficiency_util.ampgen_df("dcs", "k_plus", train=None)[:num]

    # Parameters determining mixing
    r_d = np.sqrt(0.5)
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=5 * pdg_params.mixing_x(),
        mixing_y=5 * pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    # Find weights determining mixing
    mixing_weights = _mixing_weights(cf_df, dcs_df, r_d, params, q_p)

    # Get rid of some events according to an efficiency model
    rng = np.random.default_rng(seed=0)
    dcs_efficiency_mask = _toy_efficiency_mask(
        rng, dcs_df, (efficiency_definitions.MIN_TIME, 1.0, 2.5, 1.0, 2.5, 1)
    )
    cf_efficiency_mask = _toy_efficiency_mask(
        rng, cf_df, (efficiency_definitions.MIN_TIME, 1.0, 2.5, 1.0, 2.4, 1.001)
    )

    # Train a reweighter to correct for the efficiency
    dcs_reweighter = _reweighter(dcs_df, dcs_df[dcs_efficiency_mask])
    cf_reweighter = _reweighter(cf_df, cf_df[cf_efficiency_mask])

    # Plot projection of the reweighting with and without mixing weights added in
    dcs_eff_wt = dcs_reweighter.weights(
        np.column_stack(
            (
                helicity_param(*efficiency_util.k_3pi(dcs_df[dcs_efficiency_mask])),
                dcs_df[dcs_efficiency_mask]["time"],
            )
        )
    )
    _projections(
        dcs_df,
        dcs_df[dcs_efficiency_mask],
        dcs_eff_wt,
        "toy_dcs_proj_nomix.png",
        mixing_wt=None,
    )

    # Find what the "true" Z is before the efficiency function, after mixing
    # This is the Z value that we should recover with a fit
    true_z = _z(
        dcs_df, mixing_weights
    )  # TODO is this right? feels wrong that Z only comes from DCS

    # Plot the ratios of times before mixing, after mixing, after efficiency + after correction
    dcs_test = ~dcs_df["train"]
    cf_test = ~cf_df["train"]

    dcs_eff_wt = dcs_reweighter.weights(
        np.column_stack(
            (
                helicity_param(
                    *efficiency_util.k_3pi(
                        dcs_df[dcs_test][dcs_efficiency_mask[dcs_test]]
                    )
                ),
                dcs_df[dcs_test][dcs_efficiency_mask[dcs_test]]["time"],
            )
        )
    )
    cf_eff_wt = cf_reweighter.weights(
        np.column_stack(
            (
                helicity_param(
                    *efficiency_util.k_3pi(cf_df[cf_test][cf_efficiency_mask[cf_test]])
                ),
                cf_df[cf_test][cf_efficiency_mask[cf_test]]["time"],
            )
        )
    )

    # Scale weights so we have the right fraction afterwards
    # idk if this is right
    dcs_eff_wt *= np.sum(cf_efficiency_mask) / np.sum(dcs_efficiency_mask)

    _plot_ratios(
        dcs_df[dcs_test]["time"],
        cf_df[~cf_df["train"]]["time"],
        dcs_efficiency_mask[dcs_test],
        cf_efficiency_mask[cf_test],
        mixing_weights[dcs_test],
        dcs_eff_wt,
        cf_eff_wt,
        (r_d, true_z, params),
    )

    _scan(
        dcs_df[dcs_test]["time"],
        cf_df[~cf_df["train"]]["time"],
        dcs_efficiency_mask[dcs_test],
        cf_efficiency_mask[cf_test],
        mixing_weights[dcs_test],
        dcs_eff_wt,
        cf_eff_wt,
        (r_d, true_z, params),
    )


if __name__ == "__main__":
    main()
