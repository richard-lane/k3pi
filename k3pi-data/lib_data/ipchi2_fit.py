"""
Utilities and stuff for fitting to the D0 IPCHI2 distributions

For assessing the secondary systematic

"""
import sys
import pathlib
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from iminuit import Minuit
from iminuit.util import make_func_code

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
from . import stats
from libFit import pdfs


def domain() -> Tuple[float, float]:
    """
    Domain for PDF

    """
    return -8.0, 12.0


def norm_peak(
    x: np.ndarray,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
) -> np.ndarray:
    """
    Signal model normalised over the domain for the IPCHI2 fit

    """
    left_args = (centre, width_l, alpha_l, beta)
    right_args = (centre, width_r, alpha_r, beta)
    low, high = domain()

    area = (
        quad(pdfs.signal_base, low, centre, args=left_args, points=(centre,))[0]
        + quad(pdfs.signal_base, centre, high, args=right_args, points=(centre,))[0]
    )

    if np.isnan(area):
        print(centre, width_l, width_r, alpha_l, alpha_r, beta)

    return pdfs.signal(x, centre, width_l, width_r, alpha_l, alpha_r, beta) / area


def model(
    x: np.ndarray,
    n_sig: float,
    n_bkg: float,
    centre_sig: float,
    width_l_sig: float,
    width_r_sig: float,
    alpha_l_sig: float,
    alpha_r_sig: float,
    beta_sig: float,
    centre_bkg: float,
    width_l_bkg: float,
    width_r_bkg: float,
    alpha_l_bkg: float,
    alpha_r_bkg: float,
    beta_bkg: float,
) -> np.ndarray:
    """
    Double Cruijff model thing for fitting log(D0 IPCHI2) distributions

    """
    return n_sig * norm_peak(
        x, centre_sig, width_l_sig, width_r_sig, alpha_l_sig, alpha_r_sig, beta_sig
    ) + n_bkg * norm_peak(
        x, centre_bkg, width_l_bkg, width_r_bkg, alpha_l_bkg, alpha_r_bkg, beta_bkg
    )


class BinnedChi2:
    """
    Cost function for binned fit to IPCHI2

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, counts: np.ndarray, bins: np.ndarray, error: np.ndarray = None):
        """
        Poisson errors assumed unless specified otherwise

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "n_sig",
                "n_bkg",
                "centre_sig",
                "width_l_sig",
                "width_r_sig",
                "alpha_l_sig",
                "alpha_r_sig",
                "beta_sig",
                "centre_bkg",
                "width_l_bkg",
                "width_r_bkg",
                "alpha_l_bkg",
                "alpha_r_bkg",
                "beta_bkg",
            ]
        )

        if error is None:
            error = np.sqrt(counts)

        self._counts = counts
        self._error = error
        self._bins = bins

    def __call__(
        self,
        n_sig,
        n_bkg,
        centre_sig,
        width_l_sig,
        width_r_sig,
        alpha_l_sig,
        alpha_r_sig,
        beta_sig,
        centre_bkg,
        width_l_bkg,
        width_r_bkg,
        alpha_l_bkg,
        alpha_r_bkg,
        beta_bkg,
    ):
        predicted = stats.areas(
            self._bins,
            model(
                self._bins,
                n_sig,
                n_bkg,
                centre_sig,
                width_l_sig,
                width_r_sig,
                alpha_l_sig,
                alpha_r_sig,
                beta_sig,
                centre_bkg,
                width_l_bkg,
                width_r_bkg,
                alpha_l_bkg,
                alpha_r_bkg,
                beta_bkg,
            ),
        )

        return np.sum((self._counts - predicted) ** 2 / self._error**2)


def fit(
    counts: np.ndarray,
    bins: np.ndarray,
    signal_frac_guess: float,
    sig_defaults: dict,
    bkg_defaults: dict,
    *,
    errors: np.ndarray = None,
) -> Minuit:
    """
    Perform a binned fit, return the fitter

    Beta params are fixed to the values passed in

    :param counts: array of log D0IPCHI2
    :param bins: binning used
    :param signal_frac_guess: initial guess at the signal fraction
    :param sig_defaults: dict of initial guesses
    :param bkg_defaults: dict of initial guesses
    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    assert 0.0 <= signal_frac_guess <= 1.0

    # Check for zeros since this breaks stuff
    assert not np.any(counts == 0.0), f"{counts=}"
    if errors is not None:
        assert not np.any(errors == 0.0), f"{errors=}"

    # Defaults
    total = np.sum(counts)
    n_sig = total * signal_frac_guess
    n_bkg = total * (1 - signal_frac_guess)

    chi2 = BinnedChi2(counts, bins, errors)

    fitter = Minuit(chi2, n_sig=n_sig, n_bkg=n_bkg, **sig_defaults, **bkg_defaults)

    # Limits
    sig_limits = {
        "centre_sig": (-1.5, 1.5),
        "width_l_sig": (0.5, 3.0),
        "width_r_sig": (0.5, 3.0),
        "alpha_l_sig": (0.0, 2.0),
        "alpha_r_sig": (0.0, 2.0),
        "beta_sig": (None, None),
    }
    bkg_limits = {
        "centre_bkg": (2.5, 8.0),
        "width_l_bkg": (0.5, 4.0),
        "width_r_bkg": (0.5, 4.0),
        "alpha_l_bkg": (0.0, 2.0),
        "alpha_r_bkg": (0.0, 2.0),
        "beta_bkg": (None, None),
    }

    for lims in sig_limits, bkg_limits:
        for name, vals in lims.items():
            fitter.limits[name] = vals

    fitter.limits["n_sig"] = (0.0, total)
    fitter.limits["n_bkg"] = (0.0, total)

    fitter.fixed["beta_sig"] = True
    fitter.fixed["beta_bkg"] = True

    fitter.migrad()

    return fitter


def plot(
    axes: Tuple[plt.Axes, plt.Axes],
    bins: np.ndarray,
    counts: np.ndarray,
    errs: np.ndarray,
    fit_params: np.ndarray,
) -> None:
    """
    Plot a fit and pulls on axes

    """
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Plot histogram
    err_kw = {"fmt": "k.", "elinewidth": 0.5, "markersize": 1.0}
    axes[0].errorbar(
        centres,
        counts / bin_widths,
        xerr=bin_widths / 2,
        yerr=errs / bin_widths,
        **err_kw,
    )

    predicted = stats.areas(bins, model(bins, *fit_params)) / bin_widths

    predicted_sig = (
        fit_params[0]
        * stats.areas(bins, norm_peak(bins, *fit_params[2:8]))
        / bin_widths
    )
    predicted_bkg = (
        fit_params[1] * stats.areas(bins, norm_peak(bins, *fit_params[8:])) / bin_widths
    )

    axes[0].plot(centres, predicted)
    axes[0].plot(
        centres,
        predicted_sig,
        label="signal",
    )
    axes[0].plot(
        centres,
        predicted_bkg,
        label="bkg",
    )

    # Plot pull
    diff = (counts / bin_widths) - predicted
    axes[1].plot(domain(), [1, 1], "r-")
    axes[1].errorbar(
        centres,
        diff,
        yerr=errs / bin_widths,
        **err_kw,
    )
