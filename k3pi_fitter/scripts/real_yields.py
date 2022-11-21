"""
Read the (WS) real data dataframe,
apply BDT cut to it,
apply efficiency correction,
do the mass fits,
take the yields,
plot their ratios

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars
from libFit import fit, pdfs
from lib_efficiency import efficiency_model
from lib_efficiency.efficiency_util import k_3pi
from lib_efficiency.metrics import _counts
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_time_fit import util, fitter, plotting
from lib_time_fit.definitions import TIME_BINS


def _time_bin_indices(times: np.ndarray) -> np.ndarray:
    """Time bin indices"""
    bins = (MIN_TIME, *TIME_BINS[2:-1])
    return np.digitize(times, bins)


def _bdt_cut(dataframe: pd.DataFrame, year: str, magnetisation: str) -> pd.DataFrame:
    """
    Perform the BDT cut on a dataframe and return a slice
    that passed the cut

    """
    # Always use the DCS BDT to do cuts
    clf = get_clf(year, "dcs", magnetisation)
    training_labels = list(training_vars.training_var_names())
    threshhold = 0.185
    predicted_signal = clf.predict_proba(dataframe[training_labels])[:, 1] > threshhold

    print(f"{np.sum(predicted_signal)} of {len(predicted_signal)} predicted signal")
    return dataframe[predicted_signal]


def _efficiency_wts(
    dataframe: pd.DataFrame, year: str, sign: str, magnetisation: str
) -> np.ndarray:
    """
    Weights to account for efficiency effects

    uses reweighter trained on both

    """
    # Find weights using the API
    return efficiency_model.weights(
        *k_3pi(dataframe),
        dataframe["time"],
        "both",
        year,
        sign,
        magnetisation,
        fit=False,
        cut=True,
        verbose=True,
    )


def _get_ratio(
    dcs_deltam: np.ndarray,
    cf_deltam: np.ndarray,
    mass_bins: np.ndarray,
    time_bin: int,
    dcs_weights: np.ndarray = None,
    cf_weights: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Get DCS/CF ratio and error from a (slice of) a dataframe

    """
    # Get mass counts and errors
    dcs_count, dcs_err = _counts(dcs_deltam, dcs_weights, mass_bins)
    cf_count, cf_err = _counts(cf_deltam, cf_weights, mass_bins)

    # Get yields
    (cf_yield, dcs_yield), (cf_err, dcs_err) = fit.yields(
        cf_count, dcs_count, mass_bins, time_bin, cf_err, dcs_err
    )

    # return ratio
    ratio = dcs_yield / cf_yield
    err = ratio * np.sqrt((dcs_err / dcs_yield) ** 2 + (cf_err / cf_yield) ** 2)

    return ratio, err


def _plot_scan(axis, fit_axis, time_bins, ratios, errs):
    """
    Plot a scan on an axis
    """
    n_re, n_im = 21, 20
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(0.0055, 0.0039183, 0.0065139, re_z, im_z)
                scan = fitter.scan_fit(
                    ratios,
                    errs,
                    time_bins,
                    these_params,
                    (0.0011489, 0.00064945),
                    -0.301,
                )

                vals = scan.values
                fit_vals = util.ScanParams(
                    r_d=vals[0], x=vals[1], y=vals[2], re_z=re_z, im_z=im_z
                )
                plotting.scan_fit(
                    fit_axis, fit_vals, fmt="k--", label=None, plot_kw={"alpha": 0.01}
                )

                chi2s[j, i] = scan.fval

    # fit_axis.set_ylim(0, 0.009)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    n_contours = 4
    contours = plotting.scan(
        axis, allowed_rez, allowed_imz, chi2s, levels=np.arange(n_contours)
    )

    # Plot the best fit value
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    axis.plot(allowed_rez[min_re], allowed_imz[min_im], "r*")

    return contours


def _plot_ratio(
    axis: plt.Axes,
    scan_ax: plt.Axes,
    mass_bins: np.ndarray,
    dcs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    dcs_indices: np.ndarray,
    cf_indices: np.ndarray,
    plot_kw: dict,
    dcs_weights: np.ndarray = None,
    cf_weights: np.ndarray = None,
):
    """
    Get yields from a fit, plot their ratio on an axis

    """
    if dcs_weights is None:
        dcs_weights = np.ones(len(dcs_df), dtype=np.float32)
    if cf_weights is None:
        cf_weights = np.ones(len(cf_df), dtype=np.float32)

    dcs_delta_m = dcs_df["D* mass"] - dcs_df["D0 mass"]
    cf_delta_m = cf_df["D* mass"] - cf_df["D0 mass"]

    ratios, errs = [], []
    # don't care about underflow or the last bin
    # should really code this in terms of len(TIME_BINS)
    for index in tqdm(np.unique(dcs_indices)[1:-1]):
        # slice according to time bin
        this_dcs = dcs_delta_m[dcs_indices == index]
        this_cf = cf_delta_m[cf_indices == index]

        this_dcs_wt = dcs_weights[dcs_indices == index]
        this_cf_wt = cf_weights[cf_indices == index]

        # Get the counts from the mass fit
        ratio, err = _get_ratio(
            this_dcs, this_cf, mass_bins, index, this_dcs_wt, this_cf_wt
        )

        # Append to a list or something
        ratios.append(ratio)
        errs.append(err)

    time_bins = np.array((MIN_TIME, *TIME_BINS[2:-1]))
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    widths = (time_bins[1:] - time_bins[:-1]) / 2

    axis.errorbar(centres, ratios, xerr=widths, yerr=errs, **plot_kw)
    abc_fitter = fitter.no_constraints(
        ratios, errs, time_bins, util.MixingParams(0.005, 0.0002, 0)
    )
    print(f"{plot_kw['label']}: {abc_fitter.valid=}")
    plotting.no_constraints(
        axis,
        abc_fitter.values,
        fmt=plot_kw["fmt"][0] + "--",
        label=plot_kw["label"] + " unconstrained",
    )

    constrained_fitter = fitter.constraints(
        ratios,
        errs,
        time_bins,
        util.ConstraintParams(0.0055, 0.0002, 0.0039183, 0.0065139),
        (0.0011489, 0.00064945),
        -0.301,
    )
    print(f"{plot_kw['label']}: {constrained_fitter.valid=}")
    plotting.constraints(
        axis,
        util.ConstraintParams(*constrained_fitter.values),
        fmt=plot_kw["fmt"][0] + ":",
        label=plot_kw["label"] + " constrained",
    )
    print(constrained_fitter.values)

    # return contours
    return _plot_scan(scan_ax, axis, time_bins, ratios, errs)


def _make_plots(
    dcs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    dcs_bdt_cut: pd.DataFrame,
    cf_bdt_cut: pd.DataFrame,
    dcs_weights: np.ndarray,
    cf_weights: np.ndarray,
    dcs_indices: np.ndarray,
    cf_indices: np.ndarray,
    dcs_cut_indices: np.ndarray,
    cf_cut_indices: np.ndarray,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

    """
    mass_bins = np.linspace(*pdfs.domain(), 200)
    fig, axis = plt.subplots()

    scan_fig, scan_axes = plt.subplots(1, 3, figsize=(15, 5))

    # Before anything
    contours = _plot_ratio(
        axis,
        scan_axes[0],
        mass_bins,
        dcs_df,
        cf_df,
        dcs_indices,
        cf_indices,
        {"fmt": "r.", "label": "Raw"},
    )

    # After BDT cut
    _plot_ratio(
        axis,
        scan_axes[1],
        mass_bins,
        dcs_bdt_cut,
        cf_bdt_cut,
        dcs_cut_indices,
        cf_cut_indices,
        {"fmt": "b.", "label": "After BDT"},
    )

    # After efficiency correction
    _plot_ratio(
        axis,
        scan_axes[2],
        mass_bins,
        dcs_bdt_cut,
        cf_bdt_cut,
        dcs_cut_indices,
        cf_cut_indices,
        {"fmt": "g.", "label": "After Efficiency"},
        dcs_weights,
        cf_weights,
    )

    axis.legend()
    axis.set_xlabel(r"t / $\tau$")
    axis.set_ylabel(r"$\frac{WS}{RS}$")

    fig.savefig("real_yields.png")

    scan_fig.subplots_adjust(right=0.8)
    cbar_ax = scan_fig.add_axes([0.85, 0.1, 0.05, 0.8])
    scan_fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")
    for axis, title in zip(scan_axes, ("Raw", "After BDT cut", "After Efficiency")):
        axis.set_title(title)
    scan_fig.savefig("real_scan.png")

    plt.close(fig)
    plt.close(scan_fig)


def main():
    """
    Plot a real data mass fit after all my manipulations

    """
    # Read data
    year, magnetisation = "2018", "magdown"
    dcs_df = pd.concat(get.data(year, "dcs", magnetisation))
    cf_df = pd.concat(get.data(year, "cf", magnetisation))

    dcs_df = dcs_df[: len(dcs_df) // 2]
    cf_df = cf_df[: len(cf_df) // 2]

    # Find time bin indices
    dcs_indices = _time_bin_indices(dcs_df["time"])
    cf_indices = _time_bin_indices(cf_df["time"])

    # Do BDT cut with the right threshhold
    print("Doing BDT cut")
    dcs_bdt_cut = _bdt_cut(dcs_df, year, magnetisation)
    cf_bdt_cut = _bdt_cut(cf_df, year, magnetisation)
    dcs_cut_indices = _time_bin_indices(dcs_bdt_cut["time"])
    cf_cut_indices = _time_bin_indices(cf_bdt_cut["time"])

    # Do efficiency correction
    dcs_weights = _efficiency_wts(dcs_bdt_cut, year, "dcs", magnetisation)
    cf_weights = _efficiency_wts(cf_bdt_cut, year, "cf", magnetisation)

    # Plot the yields
    _make_plots(
        dcs_df,
        cf_df,
        dcs_bdt_cut,
        cf_bdt_cut,
        dcs_weights,
        cf_weights,
        dcs_indices,
        cf_indices,
        dcs_cut_indices,
        cf_cut_indices,
    )


if __name__ == "__main__":
    main()
