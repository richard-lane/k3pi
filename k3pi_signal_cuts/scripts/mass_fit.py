"""
Do a mass fit to the real data (before BDT cuts) to estimate the amount of signal
and background that we expect in our WS sample

This requires some of the real data dataframes to exist

"""
import sys
import pathlib
import argparse
from typing import Tuple, Iterable, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from libFit import plotting, definitions, pdfs
from libFit.fit import binned_simultaneous_fit
from libFit import util
from lib_data import get, stats, cuts
from lib_time_fit.definitions import TIME_BINS


def _ws_bkg(
    domain: np.ndarray,
    scale: float,
    signal_fraction: float,
    bkg_params: Tuple[float, float],
) -> np.ndarray:
    """
    Fitted background component for WS data

    """
    return (
        scale
        * (1 - signal_fraction)
        * pdfs.normalised_bkg(domain, *bkg_params, pdfs.domain())
    )


def _rs_signal(
    domain: np.ndarray,
    scale: float,
    signal_fraction: float,
    sig_params: Tuple[float, ...],
) -> np.ndarray:
    """
    Fitted signal component for RS data, scaled using the amplitude ratio ^ 2

    """
    return (
        scale
        * signal_fraction
        * pdfs.normalised_signal(domain, *sig_params[1:], pdfs.domain())
    )


def _rs_plot(
    axes: Tuple[plt.Axes, plt.Axes],
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
    signal_region: Tuple,
):
    """
    Plot RS stuff - the signal, bkg and shaded/scaled signal on an axis

    """
    _plot(axes, delta_m, bins, params, scale)

    # Shade the RS signal region; this is where we will take the number of signal events
    # for optimising the classifier
    axes[0].fill_between(
        signal_region,
        _rs_signal(signal_region, scale, params[0], params[1:-2]),
        color="b",
        alpha=0.2,
    )


def _ws_plot(
    axes: Tuple[plt.Axes, plt.Axes],
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
    signal_region: Tuple,
):
    """
    Plot WS stuff - the signal, bkg and shaded/scaled signal on an axis

    """
    _plot(axes, delta_m, bins, params, scale)

    # Shade the WS bkg region where we want to take the number of bkg events from for optimising
    # the classifier
    axes[0].fill_between(
        signal_region,
        _ws_bkg(signal_region, scale, params[0], params[-2:]),
        color="r",
        alpha=0.2,
    )


def _shade_plot(
    axis: plt.Axes, fcn: Callable, n_total: int, bin_width: float, **kwargs
) -> None:
    """
    Given a normalised unary callable, scale it up to have an area of n_tot
    and plot shading between it and 0 on an axis

    """
    # Define the signal region
    signal_region = (144.0, 147.0)

    # Evaluate the PDF along this region and scale up
    pts = np.linspace(*signal_region, 1000)
    f_eval = bin_width * n_total * fcn(pts)

    # Plot
    axis.fill_between(
        pts,
        f_eval,
        **kwargs,
    )


def _plot_fit(
    rs: np.ndarray, ws: np.ndarray, params: Tuple, bins: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a fit

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    centres = (bins[1:] + bins[:-1]) / 2.0
    widths = bins[1:] - bins[:-1]

    # Plot the histograms
    for axis, data in zip(axes, (rs, ws)):
        axis.errorbar(centres, data, xerr=widths, yerr=np.sqrt(data), fmt="k+")

    # Plot the shaded regions
    rs_params, ws_params = util.rs_ws_params(params)
    # Signal peak
    sig_fcn = lambda pts: pdfs.normalised_signal(
        pts, *rs_params[2:-2], pdfs.reduced_domain()
    )
    _shade_plot(
        axes[0],
        sig_fcn,
        params[0],
        widths[-1],
        alpha=0.5,
        color="b",
    )

    # bkg for WS
    bkg_fcn = lambda pts: pdfs.normalised_bkg(
        pts, *ws_params[-2:], pdfs.reduced_domain()
    )
    _shade_plot(
        axes[1],
        bkg_fcn,
        params[3],
        widths[-1],
        alpha=0.5,
        color="r",
    )

    # Signal peak, scaled, on top of bkg
    amplitude_ratio = 0.055
    fcn = lambda pts: amplitude_ratio**2 * params[0] * sig_fcn(pts) + params[
        3
    ] * bkg_fcn(pts)

    sig_region = 144.0, 147.0
    pts = np.linspace(*sig_region, 1000)
    axes[1].fill_between(
        pts, widths[-1] * fcn(pts), widths[-1] * bkg_fcn(pts), alpha=0.5, color="b"
    )

    # Draw an arrow between the signal peaks to show it's scaled
    start = 145.5, 0.5 * axes[0].get_ylim()[1]
    end = (start[0], 0.5 * axes[1].get_ylim()[1])
    connector = ConnectionPatch(
        xyA=start,
        xyB=end,
        coordsA="data",
        coordsB="data",
        axesA=axes[0],
        axesB=axes[1],
        color="b",
        linewidth=2,
        arrowstyle="->",
    )
    fig.add_artist(connector)

    axes[0].set_title(r"RS $\Delta M$")
    axes[1].set_title(r"WS $\Delta M$")

    # Find the centre of the arrow in terms of fig pixels
    # xy is the mean of the start/end points as returned by ax[1].transData.transform, etc
    # For some reason calling these refreshes the canvas in the right way to make the calculation
    # work
    _ = axes[0].get_xlim(), axes[0].get_ylim()
    _ = axes[1].get_xlim(), axes[1].get_ylim()

    def data2fig(fig, ax, point):
        transform = ax.transData + fig.transFigure.inverted()
        return transform.transform(point)

    # Don't want the label exactly at the midpoint because it looks messy
    # This should be a function really but I'm quite tired now
    arrow_start, arrow_end = data2fig(fig, axes[0], start), data2fig(fig, axes[1], end)
    length = np.linalg.norm(arrow_end - arrow_start)
    dirn = (arrow_end - arrow_start) / length
    text_locn = arrow_start + 0.85 * length * dirn

    plt.text(*text_locn, rf"$\times{amplitude_ratio:.3f}^2$", transform=fig.transFigure)

    # Find the areas of the shaded bits
    n_signal = np.sum(
        pdfs.bin_areas(
            lambda pts: amplitude_ratio**2 * params[0] * sig_fcn(pts),
            np.linspace(*sig_region),
        )
    )
    n_bkg = np.sum(
        pdfs.bin_areas(
            lambda pts: params[3] * bkg_fcn(pts), np.linspace(*sig_region)
        )
    )
    print(f"{n_signal=:.4f}, {n_bkg=:.4f}")
    fig.suptitle(f"signal fraction {n_signal / (n_signal + n_bkg):.4f}")
    axes[1].set_title(f"{axes[1].get_title()}; sig/bkg {n_signal:.1f}/{n_bkg:.1f}")

    return fig, axes


def _dataframes(year: str, sign: str, magnetisation: str) -> Iterable[pd.DataFrame]:
    """
    Get the dataframes for the mass ft

    """
    low_t, high_t = TIME_BINS[2:4]
    phsp_bin = 0

    return get.binned_generator(
        get.time_binned_generator(
            cuts.cands_cut_dfs(
                cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation))
            ),
            low_t,
            high_t,
        ),
        phsp_bin,
    )


def main(*, year: str, magnetisation: str):
    """
    Create plots for the mass fits in the

    """
    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], high), (n_underflow, 150)
    )

    # Get the data in phsp bin 0 and time bin 1
    rs_count, rs_err = stats.counts_generator(
        util.delta_m_generator(_dataframes(year, "cf", magnetisation)), mass_bins
    )
    ws_count, ws_err = stats.counts_generator(
        util.delta_m_generator(_dataframes(year, "dcs", magnetisation)), mass_bins
    )

    rs_total = np.sum(rs_count)
    ws_total = np.sum(ws_count)
    initial_guess = (
        rs_total * 0.9,
        rs_total * 0.1,
        ws_total * 0.05,
        ws_total * 0.95,
        *util.signal_param_guess(2),
        *util.sqrt_bkg_param_guess("cf"),
        *util.sqrt_bkg_param_guess("dcs"),
    )
    fitter = binned_simultaneous_fit(
        rs_count[n_underflow:],
        ws_count[n_underflow:],
        mass_bins[n_underflow:],
        initial_guess,
        (mass_bins[n_underflow], mass_bins[-1]),
        rs_errors=rs_err[n_underflow:],
        ws_errors=ws_err[n_underflow:],
    )
    print(fitter)
    fig, _ = _plot_fit(rs_count, ws_count, fitter.values, mass_bins)

    fig.tight_layout()

    path = f"bdt_mass_fit_{year}_{magnetisation}.png"
    print(f"plotting {path}")
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do a mass fit in time bin 0, phsp bin 1 to assess how much sig/bkg there is"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )
    main(**vars(parser.parse_args()))
