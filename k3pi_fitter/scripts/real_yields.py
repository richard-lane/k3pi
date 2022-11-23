"""
Read the (WS) real data dataframe,
apply BDT cut to it,
apply efficiency correction,
do the mass fits,
take the yields,
plot their ratios

"""
import os
import sys
import pathlib
from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_data import get, stats
from libFit import fit, pdfs, util as mass_util
from lib_time_fit import util, fitter, plotting
from lib_time_fit.definitions import TIME_BINS


def _scan_kw(n_levels: int) -> dict:
    """
    kwargs for scan

    """
    return {
        "levels": np.arange(n_levels),
        "plot_kw": {
            "colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_levels]
        },
    }


def _plot_fits(
    axis: plt.Axes, fit_vals: np.ndarray, chi2s: np.ndarray, n_levels: int
) -> None:
    """
    Plot a scan of fits on an axis, colour-coded according to the chi2 of each

    """
    colours = _scan_kw(n_levels)["plot_kw"]["colors"]
    for params, chi2 in zip(fit_vals.ravel(), chi2s.ravel()):
        if chi2 < 1.0:
            colour = colours[0]
            alpha = 0.3
        elif chi2 < 2.0:
            colour = colours[1]
            alpha = 0.2
        elif chi2 < 3.0:
            colour = colours[2]
            alpha = 0.05
        else:
            colour = "k"
            alpha = 0.005

        plotting.scan_fit(
            axis,
            params,
            fmt=f"--",
            label=None,
            plot_kw={"alpha": alpha, "color": colour},
        )


def _plot_scan(
    axis: plt.Axes,
    fit_axis: plt.Axes,
    time_bins: np.ndarray,
    ratios: np.ndarray,
    errs: np.ndarray,
):
    """
    Plot a scan on an axis
    """
    n_re, n_im = 31, 30
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf
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
                fit_params[j, i] = util.ScanParams(
                    r_d=vals[0], x=vals[1], y=vals[2], re_z=re_z, im_z=im_z
                )

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Plot fits on the fit axis
    n_contours = 4
    _plot_fits(fit_axis, fit_params, chi2s, n_contours)

    contours = plotting.scan(
        axis, allowed_rez, allowed_imz, chi2s, **_scan_kw(n_contours)
    )

    # Plot the best fit value
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    axis.plot(allowed_rez[min_re], allowed_imz[min_im], "r*")

    return contours


def _plot_ratio(
    axis: plt.Axes,
    time_bins: np.ndarray,
    ratio: np.ndarray,
    err: np.ndarray,
) -> None:
    """
    Plot DCS/CF ratio on an axis

    """
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    widths = (time_bins[1:] - time_bins[:-1]) / 2
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.", markersize=0.1)


def _make_plots(
    dcs_counts: List[Tuple[np.ndarray, np.ndarray]],
    cf_counts: List[Tuple[np.ndarray, np.ndarray]],
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
    output_dir: str,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

    """
    dcs_yields = []
    cf_yields = []
    dcs_errs = []
    cf_errs = []
    # Do mass fits in each bin, save the yields and errors
    # Don't want under/overflow bins
    for time_bin, (dcs_count, cf_count) in enumerate(
        zip(dcs_counts[1:-1], cf_counts[1:-1])
    ):
        ((rs_yield, ws_yield), (rs_err, ws_err)) = fit.yields(
            cf_count, dcs_count, mass_bins, time_bin
        )

        cf_yields.append(rs_yield)
        dcs_yields.append(ws_yield)
        cf_errs.append(rs_err)
        dcs_errs.append(ws_err)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Do a scan, plot on an axis
    # This also plots the scan fits on the ratio axis
    ratio, err = util.ratio_err(
        np.array(dcs_yields), np.array(cf_yields), np.array(dcs_errs), np.array(cf_errs)
    )
    _plot_ratio(axes[0], time_bins[1:-1], ratio, err)
    contours = _plot_scan(axes[1], axes[0], time_bins[1:-1], ratio, err)

    # Plot the yields and errors on a different axis
    # Plot these again so that the axis limits are right
    _plot_ratio(axes[0], time_bins[1:-1], ratio, err)

    # Titles, etc.
    axes[0].set_title("fits")
    axes[1].set_title(r"$z\ \chi^2$ landscape")

    axes[0].set_xlabel(r"t/$\tau$")
    axes[0].set_ylabel(r"$\frac{WS}{RS}$")

    axes[1].set_xlabel(r"Re(Z)")
    axes[1].set_ylabel(r"Im(Z)")

    fig.suptitle("LHCb 2018 MagDown")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    fig.savefig(f"{output_dir}scan.png")


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _counts(
    year: str,
    magnetisation: str,
    bins: np.ndarray,
    time_bins: np.ndarray,
    *,
    bdt_cut: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of counts/errors in each time bin

    """
    # Get generators of time indices
    dcs_indices = _time_indices(get.data(year, "dcs", magnetisation), time_bins)
    cf_indices = _time_indices(get.data(year, "cf", magnetisation), time_bins)

    n_time_bins = len(time_bins) - 1
    dcs_counts, _ = mass_util.binned_delta_m_counts(
        get.data(year, "dcs", magnetisation), bins, n_time_bins, dcs_indices
    )
    cf_counts, _ = mass_util.binned_delta_m_counts(
        get.data(year, "cf", magnetisation), bins, n_time_bins, cf_indices
    )

    return dcs_counts, cf_counts


def main():
    """
    Plot a real data mass fit after all my manipulations

    """
    bins = np.linspace(*pdfs.domain(), 200)
    time_bins = np.array((-100, *TIME_BINS[1:], 100))

    # Get delta M values from generator of dataframes
    year, magnetisation = "2018", "magdown"

    dcs_counts, cf_counts = _counts(year, magnetisation, bins, time_bins, bdt_cut=False)

    # Plot stuff
    fit_dir = "raw_fits/"
    if not os.path.isdir(fit_dir):
        os.mkdir(fit_dir)

    _make_plots(
        dcs_counts,
        cf_counts,
        time_bins,
        bins,
        fit_dir,
    )


if __name__ == "__main__":
    main()
