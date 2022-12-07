"""
Compare scans with no corrections
to scans with BDT cut and BDT+efficiency correction

"""
import os
import sys
import pathlib
from multiprocessing import Process
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
from lib_cuts.get import cut_dfs, classifier as get_clf
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_util import k_3pi, points


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

        if chi2 < 3.0:
            plotting.scan_fit(
                axis,
                params,
                fmt="--",
                label=None,
                plot_kw={"alpha": alpha, "color": colour},
            )


def _plot_dir(bdt_cut: bool, correct_efficiency: bool, phsp_bin: int) -> str:
    """Dir to store plots in; / terminated. Creates it if it doesnt exist"""
    if bdt_cut and correct_efficiency:
        plot_dir = "eff_fits/"
    elif bdt_cut:
        plot_dir = "bdt_fits/"
    else:
        plot_dir = "raw_fits/"

    plot_dir = os.path.join(plot_dir, f"bin_{phsp_bin}/")

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    return plot_dir


def _plot_scan(
    year: str,
    magnetisation: str,
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
    *,
    bdt_cut: bool,
    correct_efficiency: bool,
    phsp_bin: int,
):
    """
    Plot a scan on an axis

    Plot also the points and the fits to them

    """
    time_bins, ratio, err = _ratio_err(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=bdt_cut,
        correct_efficiency=correct_efficiency,
        phsp_bin=phsp_bin,
    )

    # Set axis limits so that the fit plots are sensible
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_xlim(time_bins[0], 1.1 * time_bins[-1])

    n_re, n_im = 31, 30
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones((n_im, n_re), dtype=object) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(0.0055, 0.0039183, 0.0065139, re_z, im_z)
                scan = fitter.combined_fit(
                    ratio,
                    err,
                    time_bins,
                    these_params,
                    (0.0011489, 0.00064945),
                    -0.301,
                    phsp_bin,
                )

                chi2s[j, i] = scan.fval

                vals = scan.values
                fit_params[j, i] = util.ScanParams(
                    r_d=vals[0], x=vals[1], y=vals[2], re_z=re_z, im_z=im_z
                )

                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    # Plot the fits
    n_contours = 4
    _plot_fits(axes[0], fit_params, chi2s, n_contours)

    # Plot the ratios and their errors
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    widths = (time_bins[1:] - time_bins[:-1]) / 2
    axes[0].errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.", markersize=0.1)

    contours = plotting.scan(
        axes[1], allowed_rez, allowed_imz, chi2s, **_scan_kw(n_contours)
    )

    # Plot the best fit value
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    axes[1].plot(allowed_rez[min_re], allowed_imz[min_im], "r*")

    fig.suptitle(f"LHCb Unofficial {year} {magnetisation}")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    path = f"{_plot_dir(bdt_cut, correct_efficiency, phsp_bin)}scan.png"
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)


def _ratio_err(
    year: str,
    magnetisation: str,
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
    *,
    bdt_cut: bool,
    correct_efficiency: bool,
    phsp_bin: int,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

    returns time bins, ratio, err

    """
    if correct_efficiency:
        assert bdt_cut, "Cannot have efficiency without BDT cut"

    # Bin delta M
    dcs_counts, cf_counts, dcs_mass_errs, cf_mass_errs = _counts(
        year,
        magnetisation,
        mass_bins,
        time_bins,
        bdt_cut=bdt_cut,
        correct_efficiency=correct_efficiency,
        phsp_bin=phsp_bin,
    )

    # Don't want the first or last bins since they're
    # the overflows (down to -inf; up to + inf)
    # Also don't want the very last bin since the stats there suck
    time_bins = time_bins[1:-2]
    dcs_counts = dcs_counts[1:-2]
    cf_counts = cf_counts[1:-2]
    dcs_mass_errs = dcs_mass_errs[1:-2]
    cf_mass_errs = cf_mass_errs[1:-2]

    dcs_yields = []
    cf_yields = []
    dcs_errs = []
    cf_errs = []

    plot_dir = _plot_dir(bdt_cut, correct_efficiency, phsp_bin)

    # Do mass fits in each bin, save the yields and errors
    for time_bin, (dcs_count, cf_count, dcs_mass_err, cf_mass_err) in tqdm(
        enumerate(zip(dcs_counts, cf_counts, dcs_mass_errs, cf_mass_errs))
    ):
        ((rs_yield, ws_yield), (rs_err, ws_err)) = fit.yields(
            cf_count,
            dcs_count,
            mass_bins,
            time_bin,
            rs_errors=cf_mass_err,
            ws_errors=dcs_mass_err,
            path=f"{plot_dir}fit_{time_bin}.png",
        )

        cf_yields.append(rs_yield)
        dcs_yields.append(ws_yield)
        cf_errs.append(rs_err)
        dcs_errs.append(ws_err)

    ratio, err = util.ratio_err(
        np.array(dcs_yields), np.array(cf_yields), np.array(dcs_errs), np.array(cf_errs)
    )

    return time_bins, ratio, err


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _generators(
    year: str, magnetisation: str, *, bdt_cut: bool, phsp_bin: int
) -> Tuple[Iterable[pd.DataFrame], Iterable[pd.DataFrame]]:
    """
    Generator of rs/ws dataframes, with/without BDT cut

    """
    generators = [
        get.binned_generator(get.data(year, sign, magnetisation), phsp_bin)
        for sign in ("cf", "dcs")
    ]

    if bdt_cut:
        # Get the classifier
        clf = get_clf(year, "dcs", magnetisation)
        return [cut_dfs(gen, clf) for gen in generators]

    return generators


def _efficiency_generators(
    year: str, magnetisation: str, *, phsp_bin: int
) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
    """
    Generator of efficiency weights

    """
    # Do BDT cut
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=True, phsp_bin=phsp_bin)

    # Open efficiency weighters
    dcs_weighter = get_reweighter(
        year, "dcs", magnetisation, "both", fit=False, cut=True, verbose=True
    )
    cf_weighter = get_reweighter(
        year, "cf", magnetisation, "both", fit=False, cut=True, verbose=True
    )

    # Generators to get weights
    return (
        (
            cf_weighter.weights(points(*k_3pi(dataframe), dataframe["time"]))
            for dataframe in cf_gen
        ),
        (
            dcs_weighter.weights(points(*k_3pi(dataframe), dataframe["time"]))
            for dataframe in dcs_gen
        ),
    )


def _counts(
    year: str,
    magnetisation: str,
    bins: np.ndarray,
    time_bins: np.ndarray,
    *,
    bdt_cut: bool,
    correct_efficiency: bool,
    phsp_bin: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns a list of counts/errors in each time bin

    """
    # Get generators of time indices
    cf_gen, dcs_gen = _generators(
        year, magnetisation, bdt_cut=bdt_cut, phsp_bin=phsp_bin
    )

    dcs_indices = _time_indices(dcs_gen, time_bins)
    cf_indices = _time_indices(cf_gen, time_bins)

    # Need to get new generators now that we've used them up
    cf_gen, dcs_gen = _generators(
        year, magnetisation, bdt_cut=bdt_cut, phsp_bin=phsp_bin
    )

    # May need to get generators of efficiency weights too
    if correct_efficiency:
        cf_wts, dcs_wts = _efficiency_generators(year, magnetisation, phsp_bin=phsp_bin)
    else:
        cf_wts, dcs_wts = None, None

    n_time_bins = len(time_bins) - 1
    dcs_counts, dcs_errs = mass_util.binned_delta_m_counts(
        dcs_gen, bins, n_time_bins, dcs_indices, dcs_wts
    )
    cf_counts, cf_errs = mass_util.binned_delta_m_counts(
        cf_gen, bins, n_time_bins, cf_indices, cf_wts
    )

    return dcs_counts, cf_counts, dcs_errs, cf_errs


def _make_scans(
    year: str,
    magnetisation: str,
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
    phsp_bin: int,
) -> None:
    """
    Plot all 3 kinds of scans in the right phase space bins

    """
    _plot_scan(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=False,
        correct_efficiency=False,
        phsp_bin=phsp_bin,
    )

    _plot_scan(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=True,
        correct_efficiency=False,
        phsp_bin=phsp_bin,
    )

    _plot_scan(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=True,
        correct_efficiency=True,
        phsp_bin=phsp_bin,
    )


def main():
    """
    Plot raw scan, scan after BDT cut, scan after BDT cut + efficiency

    """
    year, magnetisation = "2018", "magdown"
    mass_bins = np.linspace(*pdfs.domain(), 200)
    time_bins = np.array((-np.inf, *TIME_BINS[1:], np.inf))

    procs = [
        Process(
            target=_make_scans,
            args=(year, magnetisation, time_bins, mass_bins, phsp_bin),
        )
        for phsp_bin in range(4)
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
