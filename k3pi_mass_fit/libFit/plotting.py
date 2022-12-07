"""
Plot mass fits

"""
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from . import pdfs


def mass_fit(
    axes: Tuple[plt.Axes, plt.Axes],
    counts: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    fit_params: Tuple,
) -> None:
    """
    Plot the mass fit and pulls on an axis

    :param axes: tuple of (histogram axis, pull axis)
    :param counts: counts in each bin
    :param errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_params: fit parameters; (sig frac, other params)

    """
    centres = (bins[1:] + bins[:-1]) / 2
    scale = np.sum(counts) * (bins[1] - bins[0])

    # Plot histogram
    err_kw = {"fmt": "k.", "elinewidth": 0.5, "markersize": 1.0}
    axes[0].errorbar(
        centres,
        counts,
        yerr=errs,
        **err_kw,
    )

    predicted = scale * pdfs.fractional_pdf(centres, *fit_params)
    axes[0].plot(centres, predicted)
    axes[0].plot(
        centres,
        scale * fit_params[0] * pdfs.normalised_signal(centres, *fit_params[1:-2]),
        label="signal",
    )
    axes[0].plot(
        centres,
        scale * (1 - fit_params[0]) * pdfs.normalised_bkg(centres, *fit_params[-2:]),
        label="bkg",
    )

    # Plot pull
    diff = counts - predicted
    axes[1].plot(pdfs.domain(), [1, 1], "r-")
    axes[1].errorbar(
        centres,
        diff,
        yerr=np.sqrt(counts),
        **err_kw,
    )


def simul_fits(
    rs_counts: np.ndarray,
    rs_errs: np.ndarray,
    ws_counts: np.ndarray,
    ws_errs: np.ndarray,
    bins: np.ndarray,
    fit_params: Tuple,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Plot a simultaneous RS and WS fit on four Axes - two histograms
    and two pulls

    :param rs_counts: counts in each bin
    :param rs_errs: errors on the counts in each bin
    :param ws_counts: counts in each bin
    :param ws_errs: errors on the counts in each bin
    :param bins: mass bins
    :param fit_params: fit parameters as returned by the simultaneous fitter

    :returns: the figure
    :returns: a dict of A/B/C/D and the plot axes

    """
    rs_params = (fit_params[0], *fit_params[2:])
    ws_params = tuple(fit_params[1:])

    fig, axes = plt.subplot_mosaic(
        "AAABBB\nAAABBB\nAAABBB\nCCCDDD", sharex=True, figsize=(12, 8)
    )
    mass_fit((axes["A"], axes["C"]), rs_counts, rs_errs, bins, rs_params)
    mass_fit((axes["B"], axes["D"]), ws_counts, ws_errs, bins, ws_params)

    axes["A"].legend()

    axes["C"].plot(pdfs.domain(), [1, 1], "r-")
    axes["D"].plot(pdfs.domain(), [1, 1], "r-")

    fig.tight_layout()

    return fig, axes
