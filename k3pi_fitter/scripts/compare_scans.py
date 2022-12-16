"""
Compare scans with no corrections
to scans with BDT cut and BDT+efficiency correction

"""
import os
import sys
import pathlib
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from libFit import fit, pdfs, util as mass_util
from lib_time_fit import util, fitter, plotting
from lib_time_fit.definitions import TIME_BINS
from lib_efficiency.efficiency_definitions import RS_EFF, RS_ERR, WS_EFF, WS_ERR


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
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].set_xlim(time_bins[0], 1.1 * time_bins[-1])
    axes[1, 0].set_xlim(time_bins[0], 1.1 * time_bins[-1])

    for axis, label in zip(axes[:, 0], ("LHCb Only", "LHCb + CLEO/BES")):
        axis.set_xlabel(r"t / $\tau$")
        axis.set_ylabel(f"{label}\n" r"$\frac{WS}{RS}$")

    for axis in axes[:, 1]:
        axis.set_xlabel("Re(Z)")
        axis.set_ylabel("Im(Z)")

    n_re, n_im = 31, 30
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    lhcb_chi2s = np.ones((n_im, n_re)) * np.inf
    combined_chi2s = np.ones((n_im, n_re)) * np.inf
    lhcb_params = np.ones((n_im, n_re), dtype=object) * np.inf
    combined_params = np.ones((n_im, n_re), dtype=object) * np.inf
    initial_rdxy = 0.0055, 0.0039183, 0.0065139
    xy_err = (0.0011489, 0.00064945)
    xy_corr = -0.301
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(*initial_rdxy, re_z, im_z)

                # LHCb only fit
                lhcb_fitter = fitter.scan_fit(
                    ratio,
                    err,
                    time_bins,
                    these_params,
                    xy_err,
                    xy_corr,
                )
                lhcb_chi2s[j, i] = lhcb_fitter.fval
                lhcb_vals = lhcb_fitter.values
                lhcb_params[j, i] = util.ScanParams(
                    r_d=lhcb_vals[0],
                    x=lhcb_vals[1],
                    y=lhcb_vals[2],
                    re_z=re_z,
                    im_z=im_z,
                )

                # Combined LHCb + CLEO/BES fit
                combined_fitter = fitter.combined_fit(
                    ratio,
                    err,
                    time_bins,
                    these_params,
                    xy_err,
                    xy_corr,
                    phsp_bin,
                )
                combined_chi2s[j, i] = combined_fitter.fval
                combined_vals = combined_fitter.values
                combined_params[j, i] = util.ScanParams(
                    r_d=combined_vals[0],
                    x=combined_vals[1],
                    y=combined_vals[2],
                    re_z=re_z,
                    im_z=im_z,
                )

                pbar.update(1)

    lhcb_chi2s -= np.nanmin(lhcb_chi2s)
    lhcb_chi2s = np.sqrt(lhcb_chi2s)

    combined_chi2s -= np.nanmin(combined_chi2s)
    combined_chi2s = np.sqrt(combined_chi2s)

    # Plot the fits
    n_contours = 4
    _plot_fits(axes[0, 0], lhcb_params, lhcb_chi2s, n_contours)
    _plot_fits(axes[1, 0], combined_params, combined_chi2s, n_contours)

    # Plot the ratios and their errors
    centres = (time_bins[1:] + time_bins[:-1]) / 2
    widths = (time_bins[1:] - time_bins[:-1]) / 2
    for axis in axes[:, 0]:
        axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.", markersize=0.1)

    contours = plotting.scan(
        axes[0, 1], allowed_rez, allowed_imz, lhcb_chi2s, **_scan_kw(n_contours)
    )
    plotting.scan(
        axes[1, 1], allowed_rez, allowed_imz, combined_chi2s, **_scan_kw(n_contours)
    )

    # Plot the best fit value
    for axis, chi2s in zip(axes[:, 1], (lhcb_chi2s, combined_chi2s)):
        min_im, min_re = np.unravel_index(chi2s.argmin(), chi2s.shape)
        axis.plot(allowed_rez[min_re], allowed_imz[min_im], "r*")
        axis.add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))

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
    returns time bins, ratio, err from the mass fits

    """
    if correct_efficiency:
        assert bdt_cut, "Cannot have efficiency without BDT cut"

    # Bin delta M
    dcs_counts, cf_counts, dcs_mass_errs, cf_mass_errs = mass_util.mass_counts(
        year,
        magnetisation,
        mass_bins,
        time_bins,
        bdt_cut=bdt_cut,
        correct_efficiency=correct_efficiency,
        phsp_bin=phsp_bin,
    )

    # If we're correcting for the efficiency, might want to adjust the
    # DCS counts (and their errors) to account for the absolute
    # efficiency
    # eff_ratio = WS_EFF / RS_EFF
    # eff_ratio_err = eff_ratio * np.sqrt((RS_ERR / RS_EFF) ** 2 + (WS_ERR / WS_EFF) ** 2)

    # dcs_counts *= eff_ratio
    # dcs_mass_errs = dcs_counts * np.sqrt(
    #     (dcs_mass_errs / dcs_counts) ** 2 + (eff_ratio_err / eff_ratio) ** 2
    # )

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
            # Uncomment to also plot the mass fit - for e.g. debug
            # path=f"{_plot_dir(bdt_cut, correct_efficiency, phsp_bin)}fit_{time_bin}.png",
        )

        cf_yields.append(rs_yield)
        dcs_yields.append(ws_yield)
        cf_errs.append(rs_err)
        dcs_errs.append(ws_err)

    ratio, err = util.ratio_err(
        np.array(dcs_yields), np.array(cf_yields), np.array(dcs_errs), np.array(cf_errs)
    )

    return time_bins, ratio, err


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
