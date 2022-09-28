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

from libFit import pdfs
from libFit.fit import simultaneous_fit
from lib_data import get


def _delta_m(data: pd.DataFrame) -> np.ndarray:
    """ D* - D0 Mass """
    return data["D* mass"] - data["D0 mass"]


def _plot(
    ax: plt.Axes,
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
    fmt: str,
):
    """
    Plot shared stuff - the fits + signal/bkg components

    """
    count, _ = np.histogram(delta_m, bins)
    err = np.sqrt(count)

    centres = (bins[1:] + bins[:-1]) / 2
    ax.errorbar(centres, count, yerr=err, fmt="k.")
    predicted = scale * pdfs.fractional_pdf(centres, *params)
    ax.plot(centres, predicted, fmt + "-")

    ax.plot(
        centres,
        scale * params[0] * pdfs.normalised_signal(centres, *params[1:-2]),
        fmt + "--",
        label="signal",
    )

    ax.plot(
        centres,
        scale * (1 - params[0]) * pdfs.normalised_bkg(centres, *params[-2:]),
        fmt + ":",
        label="bkg",
    )

    ax.set_xlabel(r"$\Delta M$ / MeV")


def _ws_bkg(
    domain: np.ndarray,
    scale: float,
    signal_fraction: float,
    bkg_params: Tuple[float, float],
) -> np.ndarray:
    """
    Fitted background component for WS data

    """
    return scale * (1 - signal_fraction) * pdfs.normalised_bkg(domain, *bkg_params)


def _rs_signal(
    domain: np.ndarray,
    scale: float,
    signal_fraction: float,
    sig_params: Tuple[float, ...],
) -> np.ndarray:
    """
    Fitted signal component for RS data, scaled using the amplitude ratio ^ 2

    """
    return scale * signal_fraction * pdfs.normalised_signal(domain, *sig_params)


def _rs_plot(
    axis: plt.Axes,
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
    signal_region: Tuple,
):
    """
    Plot RS stuff - the signal, bkg and shaded/scaled signal on an axis

    """
    _plot(axis, delta_m, bins, params, scale, "b")

    # Shade the RS signal region; this is where we will take the number of signal events
    # for optimising the classifier
    axis.fill_between(
        signal_region,
        _rs_signal(signal_region, scale, params[0], params[1:-2]),
        color="b",
        alpha=0.2,
    )


def _ws_plot(
    axis: plt.Axes,
    delta_m: np.ndarray,
    bins: np.ndarray,
    params: Tuple,
    scale: float,
    signal_region: Tuple,
):
    """
    Plot WS stuff - the signal, bkg and shaded/scaled signal on an axis

    """
    _plot(axis, delta_m, bins, params, scale, "r")

    # Shade the WS bkg region where we want to take the number of bkg events from for optimising
    # the classifier
    axis.fill_between(
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
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    signal_region = np.linspace(144, 147, 200)
    # scales assume equally spaced bins
    rs_params = params[:-1]
    ws_params = (params[-1], *params[1:-1])
    rs_scale = len(rs) * (bins[1] - bins[0])
    _rs_plot(ax[0], rs, bins, rs_params, rs_scale, signal_region)

    ws_scale = len(ws) * (bins[1] - bins[0])
    _ws_plot(
        ax[1],
        ws,
        bins,
        ws_params,
        ws_scale,
        signal_region,
    )

    # Scale and shade also the RS signal on the WS plot
    def _scaled_signal(domain):
        return amplitude_ratio ** 2 * _rs_signal(
            domain, rs_scale, rs_params[0], rs_params[1:-2]
        )

    amplitude_ratio = 0.0601387
    ax[1].fill_between(
        signal_region,
        _scaled_signal(signal_region),
        color="b",
        alpha=0.2,
    )

    ax[0].set_title(r"RS $\Delta M$")
    ax[1].set_title(r"WS $\Delta M$")

    # Draw an arrow between the signal peaks to show it's scaled
    start = 145.5, 20000
    end = (start[0], start[1] * amplitude_ratio ** 2)
    connector = ConnectionPatch(
        xyA=start,
        xyB=end,
        coordsA="data",
        coordsB="data",
        axesA=ax[0],
        axesB=ax[1],
        color="b",
        linewidth=2,
        arrowstyle="->",
    )
    fig.add_artist(connector)

    # Find the centre of the arrow in terms of fig pixels
    # xy is the mean of the start/end points as returned by ax[1].transData.transform, etc
    # For some reason calling these refreshes the canvas in the right way to make the calculation
    # work
    _ = ax[0].get_xlim(), ax[0].get_ylim()
    _ = ax[1].get_xlim(), ax[1].get_ylim()

    def data2fig(fig, ax, point):
        transform = ax.transData + fig.transFigure.inverted()
        return transform.transform(point)

    # Don't want the label exactly at the midpoint because it looks messy
    # This should be a function really but I'm quite tired now
    arrow_start, arrow_end = data2fig(fig, ax[0], start), data2fig(fig, ax[1], end)
    length = np.linalg.norm(arrow_end - arrow_start)
    dirn = (arrow_end - arrow_start) / length
    text_locn = arrow_start + 0.85 * length * dirn

    plt.text(*text_locn, fr"$\times{amplitude_ratio:.3f}^2$", transform=fig.transFigure)

    # Find the areas of the shaded bits
    factor = (bins[1] - bins[0]) / (signal_region[1] - signal_region[0])
    n_signal = factor * np.trapz(_scaled_signal(signal_region), signal_region)
    n_bkg = factor * np.trapz(
        _ws_bkg(signal_region, ws_scale, ws_params[0], ws_params[-2:]),
        signal_region,
    )

    print(f"{n_signal=:.4f}, {n_bkg=:.4f}")
    fig.suptitle(f"signal fraction {n_signal / (n_signal + n_bkg):.4f}")
    ax[1].set_title(f"{ax[1].get_title()}; sig/bkg {n_signal:.1f}/{n_bkg:.1f}")

    return fig, ax


def main():
    """
    Create plots

    """
    rs = _delta_m(pd.concat(get.data("2018", "cf", "magdown")))
    ws = _delta_m(pd.concat(get.data("2018", "dcs", "magdown")))

    fitter = simultaneous_fit(rs, ws, 5)
    fig, _ = _plot_fit(rs, ws, fitter.values)

    fig.tight_layout()

    plt.savefig("fit.png")

    plt.show()


if __name__ == "__main__":
    main()
