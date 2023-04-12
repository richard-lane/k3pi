"""
Utilities and stuff for fitting to the D0 IPCHI2 distributions

For assessing the secondary systematic

"""
import sys
import pickle
import pathlib
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from iminuit.util import make_func_code

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
from . import stats, cuts
from libFit import pdfs


class IPFitError(Exception):
    """
    Something went wrong

    """


class IPFitZeroCountsError(IPFitError):
    """
    0 Counts encountered
    """


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


def secondary_peak(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Peak for the secondary model for the IPCHI2 fit

    """
    area = quad(
        lambda x: pdfs.signal_base(x, centre, width, alpha, beta),
        *domain(),
    )[0]

    return pdfs.signal_base(x, centre, width, alpha, beta) / area


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
    width_bkg: float,
    alpha_bkg: float,
    beta_bkg: float,
) -> np.ndarray:
    """
    Cruijff model for prompt, modified Gaussian for secondary

    """
    return n_sig * norm_peak(
        x, centre_sig, width_l_sig, width_r_sig, alpha_l_sig, alpha_r_sig, beta_sig
    ) + n_bkg * secondary_peak(x, centre_bkg, width_bkg, alpha_bkg, beta_bkg)


class BinnedChi2:
    """Cost function for binned fit to IPCHI2"""

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
                "width_bkg",
                "alpha_bkg",
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
        width_bkg,
        alpha_bkg,
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
                width_bkg,
                alpha_bkg,
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
    if np.any(counts == 0.0):
        raise IPFitZeroCountsError(f"{counts=}")

    if errors is not None and np.any(errors == 0.0):
        raise IPFitZeroCountsError(f"{errors=}")

    # Defaults
    total = np.sum(counts)
    n_sig = total * signal_frac_guess
    n_bkg = total * (1 - signal_frac_guess)

    chi2 = BinnedChi2(counts, bins, errors)

    fitter = Minuit(chi2, n_sig=n_sig, n_bkg=n_bkg, **sig_defaults, **bkg_defaults)

    # Limits
    bkg_limits = {
        "centre_bkg": (2.5, 8.0),
        "width_bkg": (0.5, 3.0),
        "alpha_bkg": (0.0, 2.0),
        "beta_bkg": (None, None),
    }

    for name, vals in bkg_limits.items():
        fitter.limits[name] = vals

    fitter.limits["n_sig"] = (0.0, total)
    fitter.limits["n_bkg"] = (0.0, total)

    fitter.fixed["centre_sig"] = True
    fitter.fixed["width_l_sig"] = True
    fitter.fixed["width_r_sig"] = True
    fitter.fixed["alpha_l_sig"] = True
    fitter.fixed["alpha_r_sig"] = True
    fitter.fixed["beta_sig"] = True

    # fitter.fixed["beta_bkg"] = True

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

    predicted = (
        pdfs.bin_areas(lambda pts: model(pts, *fit_params), bins, 50) / bin_widths
    )

    predicted_sig = (
        fit_params[0]
        * pdfs.bin_areas(lambda pts: norm_peak(pts, *fit_params[2:8]), bins, 50)
        / bin_widths
    )
    predicted_bkg = (
        fit_params[1]
        * pdfs.bin_areas(lambda pts: secondary_peak(pts, *fit_params[8:]), bins, 50)
        / bin_widths
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
    diff = ((counts / bin_widths) - predicted) / (errs / bin_widths)
    axes[1].axhline(0, color="k")
    for pos in (-1, 1):
        axes[1].axhline(pos, color="r", alpha=0.5)
    axes[1].errorbar(
        centres,
        diff,
        yerr=1.0,
        **err_kw,
    )

    axes[0].set_ylabel("N/bin width")
    axes[1].set_xlabel(r"D0 IP$\chi^2$")
    axes[1].set_ylabel(r"$\sigma$")


def ext_model(
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
    width_bkg: float,
    alpha_bkg: float,
    beta_bkg: float,
) -> Tuple[float, np.ndarray]:
    """
    Cruijff model for prompt, modified Gaussian for secondary

    Also returns total count

    """
    return n_sig + n_bkg, model(
        x,
        n_sig,
        n_bkg,
        centre_sig,
        width_l_sig,
        width_r_sig,
        alpha_l_sig,
        alpha_r_sig,
        beta_sig,
        centre_bkg,
        width_bkg,
        alpha_bkg,
        beta_bkg,
    )


def unbinned_fit(
    log_ipchi2: np.ndarray,
    signal_frac_guess: float,
    sig_defaults: dict,
    bkg_defaults: dict,
) -> Minuit:
    """
    Perform an unbinned fit, return the fitter

    Beta params are fixed to the values passed in

    :param log_ipchi2: array of ipchi2 values
    :param signal_frac_guess: initial guess at the signal fraction
    :param sig_defaults: dict of initial guesses
    :param bkg_defaults: dict of initial guesses

    :returns: fitter after performing the fit

    """
    assert 0.0 <= signal_frac_guess <= 1.0

    # Defaults
    total = len(log_ipchi2)
    n_sig = total * signal_frac_guess
    n_bkg = total * (1 - signal_frac_guess)

    cost_fcn = ExtendedUnbinnedNLL(log_ipchi2, ext_model)

    fitter = Minuit(cost_fcn, n_sig=n_sig, n_bkg=n_bkg, **sig_defaults, **bkg_defaults)

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
        "width_bkg": (0.5, 4.0),
        "alpha_bkg": (0.0, 2.0),
        "beta_bkg": (None, None),
    }

    for lims in sig_limits, bkg_limits:
        for name, vals in lims.items():
            fitter.limits[name] = vals

    fitter.limits["n_sig"] = (0.0, total)
    fitter.limits["n_bkg"] = (0.0, total)

    print("initial fit")
    fitter.fixed["beta_sig"] = True
    fitter.fixed["beta_bkg"] = True
    fitter.migrad()

    print("beta fit")
    fitter.fixed["beta_sig"] = False
    fitter.fixed["beta_bkg"] = False
    fitter.migrad()

    return fitter


def sweight_fcn(params: Tuple) -> Callable:
    """
    Find a function that gives weights to project out
    the prompt events from an IP fit

    """

    # Fcns given these params
    def prompt(x):
        return norm_peak(x, *params[2:8])

    def sec(x):
        return secondary_peak(x, *params[8:])

    # Need this to be normalised
    def overall(x):
        return model(x, *params) / (params[0] + params[1])

    # Evaluate signal, bkg + both across the domain
    pts = np.linspace(*domain(), 1000)
    prompt_vals = prompt(pts)
    secondary_vals = sec(pts)
    model_vals = overall(pts)

    # Construct W matrix
    w_matrix = np.zeros((2, 2))
    w_matrix[0, 0] = np.trapz(prompt_vals**2 / model_vals, x=pts)
    w_matrix[1, 1] = np.trapz(secondary_vals**2 / model_vals, x=pts)
    w_matrix[0, 1] = w_matrix[1, 0] = np.trapz(
        prompt_vals * secondary_vals / model_vals, x=pts
    )

    # Find some params by inverting this matrix or something
    alpha = solve(w_matrix, [1, 0])

    # Construct the weighting function
    return lambda x: (alpha[0] * prompt(x) + alpha[1] * sec(x)) / overall(x)


def sec_frac_below_cut(prompt_params: dict, fit_vals: Tuple) -> float:
    """
    Find the fraction of secondaries below the cut value
    from a fit

    """
    pts = np.linspace(domain()[0], np.log(cuts.MAX_IPCHI2), 1_000_000)

    # Find prompt integral up to the cut
    prompt = stats.integral(
        pts, fit_vals[0] * norm_peak(pts, *tuple(prompt_params.values()))
    )

    # Find secondary integral up to the cut
    secondary = stats.integral(pts, fit_vals[1] * secondary_peak(pts, *fit_vals[2:]))

    return secondary / (secondary + prompt)


def ext_model_fixed_prompt(
    prompt_params: dict,
) -> Callable:
    """
    Returns a fcn that takes n_sig, n_bkg + bkg params
    that returns n tot and the model

    :param prompt_params: dict containing keys: centre, width_l, ...etc
    :param sec_params: dict containing keys: centre, width_l, ...etc

    """

    def ret_fcn(
        pts: np.ndarray,
        n_sig: float,
        n_bkg: float,
        centre: float,
        width: float,
        alpha: float,
        beta: float,
    ):
        """
        autogenerated

        """
        return (
            n_sig + n_bkg,
            n_sig * norm_peak(pts, **prompt_params)
            + n_bkg * secondary_peak(pts, centre, width, alpha, beta),
        )

    return ret_fcn


def fixed_prompt_unbinned_fit(
    log_ipchi2: np.ndarray,
    signal_frac_guess: float,
    sig_defaults: dict,
    bkg_defaults: dict,
    centre_lim: Tuple[float, float],
    width_lim: Tuple[float, float],
) -> Minuit:
    """
    Perform an unbinned fit, return the fitter

    :param log_ipchi2: array of ipchi2 values
    :param signal_frac_guess: initial guess at the signal fraction
    :param sig_defaults: dict of initial guesses - these are fixed
    :param bkg_defaults: dict of initial guesses - these are floated
    :param centre_lim: fit limits on the secondary peak centre
    :param width_lim: fit limits on the secondary peak width

    :returns: fitter after performing the fit

    """
    assert 0.0 <= signal_frac_guess <= 1.0

    # Create a fcn where our prompt params are fixed
    fit_fcn = ext_model_fixed_prompt(sig_defaults)

    # Defaults
    total = len(log_ipchi2)
    n_sig = total * signal_frac_guess
    n_bkg = total * (1 - signal_frac_guess)

    cost_fcn = ExtendedUnbinnedNLL(log_ipchi2, fit_fcn)

    fitter = Minuit(cost_fcn, n_sig=n_sig, n_bkg=n_bkg, **bkg_defaults)
    fitter.limits["centre"] = centre_lim
    fitter.limits["width"] = width_lim

    fitter.limits["alpha"] = (0.0, 2.0)
    fitter.limits["n_sig"] = (0.0, total)
    fitter.limits["n_bkg"] = (0.0, total)

    fitter.fixed["beta"] = True
    fitter.fixed["alpha"] = True

    fitter.migrad()

    return fitter


def plot_fixed_prompt(
    axes: Tuple[plt.Axes, plt.Axes],
    bins: np.ndarray,
    counts: np.ndarray,
    errs: np.ndarray,
    fit_params: np.ndarray,
    sig_params: dict,
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

    fit_fcn = lambda x: ext_model_fixed_prompt(sig_params)(x, *fit_params)[1]

    predicted = stats.areas(bins, fit_fcn(bins)) / bin_widths
    predicted_sig = (
        fit_params[0] * stats.areas(bins, norm_peak(bins, **sig_params)) / bin_widths
    )
    predicted_bkg = (
        fit_params[1]
        * stats.areas(bins, secondary_peak(bins, *fit_params[2:]))
        / bin_widths
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
    axes[0].axvline(np.log(cuts.MAX_IPCHI2))

    # Plot pull
    diff = ((counts / bin_widths) - predicted) / (errs / bin_widths)
    axes[1].axhline(0, color="k")
    for pos in (-1, 1):
        axes[1].axhline(pos, color="r", alpha=0.5)
    axes[1].errorbar(
        centres,
        diff,
        yerr=1.0,
        **err_kw,
    )

    axes[0].set_ylabel("N/bin width")
    axes[1].set_xlabel(r"D0 IP$\chi^2$")
    axes[1].set_ylabel(r"$\sigma$")


def sec_frac_lowtime_file(sign: str) -> pathlib.Path:
    """
    Location for a pickle dump holding the params
    from the low time fit

    """
    return (
        pathlib.Path(__file__).resolve().parents[1] / f"ip_fit_lowt_params_{sign}.pkl"
    )


def low_t_params(sign: str):
    """
    Get the low t fit params from pickle dump

    """
    with open(str(sec_frac_lowtime_file(sign)), "rb") as dump_f:
        return pickle.load(dump_f)


def sec_frac_file(sign: str) -> pathlib.Path:
    """
    Location for a pickle dump to hold the secondaries

    """
    return pathlib.Path(__file__).resolve().parents[1] / f"sec_frac_{sign}.pkl"


def sec_fracs(sign: str) -> np.ndarray:
    """
    Get the secondary fraction from a pickle dump

    """
    with open(str(sec_frac_file(sign)), "rb") as dump_f:
        return pickle.load(dump_f)


def correct(target: np.ndarray, secondary_fractions: np.ndarray) -> np.ndarray:
    """
    Correct an array (i.e. yield or error) by the provided secondary fractions

    """
    assert len(target) == len(secondary_fractions)

    return target * (1 - secondary_fractions)
