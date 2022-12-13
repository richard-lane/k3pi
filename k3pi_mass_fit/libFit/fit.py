"""
Function for doing the mass fit

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

from . import pdfs, plotting


def binned_fit(
    counts: np.ndarray,
    bins: np.ndarray,
    sign: str,
    time_bin: int,
    signal_frac_guess: float,
    *,
    errors: np.ndarray = None,
) -> Minuit:
    """
    Perform a binned fit, return the fitter

    :param counts: array of D* - D0 mass differences
    :param bins: delta M binning used for the fit
    :param sign: either "RS" or "WS"
    :param time_bin: which time bin we're performing the fit in; this determines the value of beta
    :param signal_frac_guess: initial guess at the signal fraction
    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert sign in {"RS", "WS"}
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a, b = pdfs.background_defaults(sign)

    chi2 = pdfs.BinnedChi2(counts, bins, errors)

    m = Minuit(
        chi2,
        signal_fraction=signal_frac_guess,
        bkg_fraction=(1 - signal_frac_guess),
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        a=a,
        b=b,
    )
    m.limits = (
        (0.0, 1.0),  # Signal fraction
        (0.0, 1.0),  # Bkg fraction
        (144.0, 147.0),  # Centre
        (0.1, 1.0),  # width L
        (0.1, 1.0),  # width R
        (0.01, 1.0),  # alpha L
        (0.01, 1.0),  # alpha R
        (None, None),  # Beta (fixed below)
        (None, None),  # Background a
        (None, None),  # Background b
    )

    m.fixed["beta"] = True

    m.migrad()

    return m


def binned_simultaneous_fit(
    rs_counts: np.ndarray,
    ws_counts: np.ndarray,
    bins: np.ndarray,
    time_bin: int,
    rs_errors: np.ndarray = None,
    ws_errors: np.ndarray = None,
) -> Minuit:
    """
    Perform the fit, return the fitter

    :param rs_delta_m: D* - D0 mass difference counts in the bins
    :param ws_delta_m: D* - D0 mass difference counts in the bins
    :param bins: delta M binning
    :param time_bin: which time bin we're performing the fit in; this determines the value of beta
    :param rs_errors: optional bin errors. Poisson errors assumed otherwise
    :param ws_errors: optional bin errors. Poisson errors assumed otherwise

    :returns: fitter after performing the fit

    """
    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a, b = pdfs.background_defaults("RS")

    chi2 = pdfs.SimultaneousBinnedChi2(rs_counts, ws_counts, bins, rs_errors, ws_errors)

    m = Minuit(
        chi2,
        rs_signal_fraction=0.5,
        rs_bkg_fraction=0.5,
        ws_signal_fraction=0.05,
        ws_bkg_fraction=0.95,
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        a=a,
        b=b,
    )
    m.limits["rs_signal_fraction"] = (0.0, 1.0)
    m.limits["rs_bkg_fraction"] = (0.0, 1.0)
    m.limits["ws_signal_fraction"] = (0.0, 1.0)
    m.limits["ws_bkg_fraction"] = (0.0, 1.0)
    m.limits["centre"] = (144.0, 147.0)
    m.limits["width_l"] = (0.1, 1.0)
    m.limits["width_r"] = (0.1, 1.0)
    m.limits["alpha_l"] = (0.01, 1.0)
    m.limits["alpha_r"] = (0.01, 1.0)

    m.fixed["beta"] = True

    m.migrad(ncall=5000)

    if m.valid:
        m.minos()
    else:
        print(m)

    return m


def yields(
    rs_counts: np.ndarray,
    ws_counts: np.ndarray,
    bins: np.ndarray,
    time_bin: int,
    rs_errors: np.ndarray = None,
    ws_errors: np.ndarray = None,
    path: str = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get RS and WS yields and their errors from a binned simultaneous mass fit

    returns ((RS yield, WS yield), (RS err, WS err))

    if path provided, also makes a plot

    """
    fitter = binned_simultaneous_fit(
        rs_counts,
        ws_counts,
        bins,
        time_bin,
        rs_errors,
        ws_errors,
    )

    fit_fracs = fitter.values[:2]
    fit_errs = fitter.errors[:2]

    totals = tuple(np.sum(counts) for counts in (rs_counts, ws_counts))

    fit_yields = (fit_frac * total for fit_frac, total in zip(fit_fracs, totals))
    fit_errs = (fit_err * total for fit_err, total in zip(fit_errs, totals))

    if path:
        if rs_errors is None:
            rs_errors = np.sqrt(rs_counts)
        if ws_errors is None:
            ws_errors = np.sqrt(ws_counts)

        fig, _ = plotting.simul_fits(
            rs_counts, rs_errors, ws_counts, ws_errors, bins, fitter.values, binned=True
        )
        fig.suptitle(f"{fitter.valid=}")
        fig.tight_layout()

        print(f"Saving {path}")
        plt.savefig(path)
        plt.close(fig)

    return fit_yields, fit_errs
