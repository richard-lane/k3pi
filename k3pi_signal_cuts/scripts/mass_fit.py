"""
Do a mass fit to the real data (before BDT cuts) to estimate the amount of signal
and background that we expect in our WS sample

This requires you/me to have cloned the k3pi_mass_fit repo in the same dir as
you cloned k3pi_signal_cuts; also requires some of the real data dataframes
to exist

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from libFit import plotting, definitions, pdfs
from libFit.fit import binned_simultaneous_fit
from libFit import util
from lib_data import get, stats


def _delta_m(data: pd.DataFrame) -> np.ndarray:
    """D* - D0 Mass"""
    return data["D* mass"] - data["D0 mass"]


def _plot(
    axes: Tuple[plt.Axes, plt.Axes],
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
):
    """
    Plot shared stuff - the fits + signal/bkg components

    """
    count, _ = np.histogram(delta_m, bins)
    err = np.sqrt(count)

    plotting.mass_fit(axes, count, err, bins, (bins[0], bins[-1]), params)


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


def _plot_fit(
    rs: np.ndarray, ws: np.ndarray, params: tuple
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a fit, assuming we used time bin number 5

    """
    bins = np.linspace(140, 152, 150)
    fig, ax = plt.subplot_mosaic("AAABBB\nAAABBB\nAAABBB\nCCCDDD", figsize=(12, 6))

    signal_region = np.linspace(144, 147, 200)
    # scales assume equally spaced bins
    rs_params, ws_params = util.rs_ws_params(params)

    rs_scale = len(rs) * (bins[1] - bins[0])
    _rs_plot((ax["A"], ax["C"]), rs, bins, rs_params, rs_scale, signal_region)

    ws_scale = len(ws) * (bins[1] - bins[0])
    _ws_plot(
        (ax["B"], ax["D"]),
        ws,
        bins,
        ws_params,
        ws_scale,
        signal_region,
    )

    # Scale and shade also the RS signal on the WS plot
    def _scaled_signal(domain):
        return amplitude_ratio**2 * _rs_signal(
            domain, rs_scale, rs_params[0], rs_params[1:-2]
        )

    amplitude_ratio = 0.0601387
    ax["A"].fill_between(
        signal_region,
        _scaled_signal(signal_region),
        color="b",
        alpha=0.2,
    )

    ax["A"].set_title(r"RS $\Delta M$")
    ax["B"].set_title(r"WS $\Delta M$")

    # Draw an arrow between the signal peaks to show it's scaled
    start = 145.5, 20000
    end = (start[0], start[1] * amplitude_ratio**2)
    connector = ConnectionPatch(
        xyA=start,
        xyB=end,
        coordsA="data",
        coordsB="data",
        axesA=ax["A"],
        axesB=ax["B"],
        color="b",
        linewidth=2,
        arrowstyle="->",
    )
    fig.add_artist(connector)

    # Find the centre of the arrow in terms of fig pixels
    # xy is the mean of the start/end points as returned by ax[1].transData.transform, etc
    # For some reason calling these refreshes the canvas in the right way to make the calculation
    # work
    _ = ax["A"].get_xlim(), ax["A"].get_ylim()
    _ = ax["B"].get_xlim(), ax["B"].get_ylim()

    def data2fig(fig, ax, point):
        transform = ax.transData + fig.transFigure.inverted()
        return transform.transform(point)

    # Don't want the label exactly at the midpoint because it looks messy
    # This should be a function really but I'm quite tired now
    arrow_start, arrow_end = data2fig(fig, ax["A"], start), data2fig(fig, ax["B"], end)
    length = np.linalg.norm(arrow_end - arrow_start)
    dirn = (arrow_end - arrow_start) / length
    text_locn = arrow_start + 0.85 * length * dirn

    plt.text(*text_locn, rf"$\times{amplitude_ratio:.3f}^2$", transform=fig.transFigure)

    # Find the areas of the shaded bits
    factor = (bins[1] - bins[0]) / (signal_region[1] - signal_region[0])
    n_signal = factor * np.trapz(_scaled_signal(signal_region), signal_region)
    n_bkg = factor * np.trapz(
        _ws_bkg(signal_region, ws_scale, ws_params[0], ws_params[-2:]),
        signal_region,
    )

    print(f"{n_signal=:.4f}, {n_bkg=:.4f}")
    fig.suptitle(f"signal fraction {n_signal / (n_signal + n_bkg):.4f}")
    ax["B"].set_title(f"{ax['B'].get_title()}; sig/bkg {n_signal:.1f}/{n_bkg:.1f}")

    return fig, ax


def main():
    """
    Create plots

    """
    bins = definitions.mass_bins()

    rs_count, rs_err = stats.counts_generator(
        (_delta_m(dataframe) for dataframe in get.data("2018", "cf", "magdown")), bins
    )
    ws_count, ws_err = stats.counts_generator(
        (_delta_m(dataframe) for dataframe in get.data("2018", "dcs", "magdown")), bins
    )

    # TODO do this fit with proper bins
    rs_total = np.sum(rs_count)
    ws_total = np.sum(ws_count)
    initial_guess = (
        rs_total * 0.9,
        rs_total * 0.1,
        ws_total * 0.05,
        ws_total * 0.95,
        *util.signal_param_guess(5),
        *util.sqrt_bkg_param_guess("cf"),
        *util.sqrt_bkg_param_guess("dcs"),
    )
    fitter = binned_simultaneous_fit(
        rs_count,
        ws_count,
        bins,
        initial_guess,
        (bins[0], bins[-1]),
        rs_errors=rs_err,
        ws_errors=ws_err,
    )
    fig, _ = _plot_fit(rs_count, ws_count, fitter.values)

    fig.tight_layout()

    plt.savefig("fit.png")

    plt.show()


if __name__ == "__main__":
    main()
