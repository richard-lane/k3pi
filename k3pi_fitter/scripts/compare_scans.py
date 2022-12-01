"""
Compare scans with no corrections
to scans with BDT cut and BDT+efficiency correction

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


def _plot_scan(
    axis: plt.Axes,
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

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    n_contours = 4
    contours = plotting.scan(
        axis, allowed_rez, allowed_imz, chi2s, **_scan_kw(n_contours)
    )

    # Plot the best fit value
    min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
    axis.plot(allowed_rez[min_re], allowed_imz[min_im], "r*")

    return contours


def _ratio_err(
    year: str,
    magnetisation: str,
    time_bins: np.ndarray,
    mass_bins: np.ndarray,
    *,
    bdt_cut: bool,
    correct_efficiency: bool,
):
    """
    Plot histograms of the masses,
    then fit to them and plot these also

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
    )

    # Don't want the first or last bins since they're too
    # sparsely populated
    time_bins = time_bins[1:-1]
    dcs_counts = dcs_counts[1:-1]
    cf_counts = cf_counts[1:-1]
    dcs_mass_errs = dcs_mass_errs[1:-1]
    cf_mass_errs = cf_mass_errs[1:-1]

    dcs_yields = []
    cf_yields = []
    dcs_errs = []
    cf_errs = []

    if bdt_cut and correct_efficiency:
        plot_dir = "eff_fits/"
    elif bdt_cut:
        plot_dir = "bdt_fits/"
    else:
        plot_dir = "raw_fits/"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # Do mass fits in each bin, save the yields and errors
    for time_bin, (dcs_count, cf_count, dcs_mass_err, cf_mass_err) in tqdm(
        enumerate(zip(dcs_counts, cf_counts, dcs_mass_errs, cf_mass_errs))
    ):
        # TODO deal with non-Poisson errors
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

    return ratio, err


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _generators(
    year: str, magnetisation: str, *, bdt_cut: bool
) -> Tuple[Iterable[pd.DataFrame], Iterable[pd.DataFrame]]:
    """
    Generator of rs/ws dataframes, with/without BDT cut

    """
    generators = get.data(year, "cf", magnetisation), get.data(
        year, "dcs", magnetisation
    )

    if bdt_cut:
        # Get the classifier
        clf = get_clf(year, "dcs", magnetisation)
        return [cut_dfs(gen, clf) for gen in generators]

    return generators


def _efficiency_generators(
    year: str, magnetisation: str
) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
    """
    Generator of efficiency weights

    """
    # Do BDT cut
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=True)

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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns a list of counts/errors in each time bin

    """
    # TODO deal with efficiency as well probably
    # Get generators of time indices
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=bdt_cut)

    dcs_indices = _time_indices(dcs_gen, time_bins)
    cf_indices = _time_indices(cf_gen, time_bins)

    # Need to get new generators now that we've used them up
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=bdt_cut)

    # May need to get generators of efficiency weights too
    if correct_efficiency:
        cf_wts, dcs_wts = _efficiency_generators(year, magnetisation)
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


def main():
    """
    Plot raw scan, scan after BDT cut, scan after BDT cut + efficiency

    """
    mass_bins = np.linspace(*pdfs.domain(), 200)
    time_bins = np.array((-np.inf, *TIME_BINS[1:], np.inf))

    # Get delta M values from generator of dataframes
    year, magnetisation = "2018", "magdown"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw
    ratio, err = _ratio_err(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=False,
        correct_efficiency=False,
    )
    # Don't want the last time bin
    # error on ratio keeps coming out NaN
    # TODO make that not happen
    contours = _plot_scan(axes[0], time_bins[1:-2], ratio[:-1], err[:-1])

    # With BDT cut
    ratio, err = _ratio_err(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=True,
        correct_efficiency=False,
    )
    # TODO
    _plot_scan(
        axes[1],
        time_bins[1:-2],
        ratio[:-1],
        err[:-1],
    )

    # With efficiency and BDT cut
    ratio, err = _ratio_err(
        year,
        magnetisation,
        time_bins,
        mass_bins,
        bdt_cut=True,
        correct_efficiency=True,
    )
    # TODO
    _plot_scan(
        axes[2],
        time_bins[1:-2],
        ratio[:-1],
        err[:-1],
    )

    # Titles, etc.
    fig.suptitle("LHCb 2018 MagDown")
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.06, 0.755])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.show()
    path = "scans.png"
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
