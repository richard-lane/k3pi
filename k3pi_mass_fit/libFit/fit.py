"""
Function for doing the mass fit

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

from . import pdfs, plotting, bkg


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
    :param sign: either "cf" or "dcs"
    :param time_bin: which time bin we're performing the fit in; this determines the value of beta
    :param signal_frac_guess: initial guess at the signal fraction
    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert sign in {"cf", "dcs"}
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a, b = pdfs.background_defaults(sign)

    chi2 = pdfs.BinnedChi2(counts, bins, errors)

    n_tot = np.sum(counts)
    fitter = Minuit(
        chi2,
        n_sig=signal_frac_guess * n_tot,
        n_bkg=(1 - signal_frac_guess) * n_tot,
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        a=a,
        b=b,
    )
    fitter.limits = (
        (0, n_tot),  # N sig
        (0, n_tot),  # N bkg
        (144.0, 147.0),  # Centre
        (0.1, 1.0),  # width L
        (0.1, 1.0),  # width R
        (0.0, 2.0),  # alpha L
        (0.0, 2.0),  # alpha R
        (None, None),  # Beta (fixed below)
        (None, None),  # Background a
        (None, None),  # Background b
    )

    fitter.fixed["beta"] = True
    fitter.migrad()

    return fitter


def binned_fit_reduced(
    counts: np.ndarray,
    bins: np.ndarray,
    sign: str,
    time_bin: int,
    signal_frac_guess: float,
    *,
    errors: np.ndarray = None,
) -> Minuit:
    """
    Perform a binned fit with reduced domain, return the fitter

    :param counts: array of D* - D0 mass differences
    :param bins: delta M binning used for the fit
    :param sign: either "cf" or "dcs"
    :param time_bin: which time bin we're performing the fit in; this determines the value of beta
    :param signal_frac_guess: initial guess at the signal fraction
    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert sign in {"cf", "dcs"}
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a, b = pdfs.background_defaults(sign)

    chi2 = pdfs.BinnedChi2Reduced(counts, bins, errors)

    n_tot = np.sum(counts)
    m = Minuit(
        chi2,
        n_sig=signal_frac_guess * n_tot,
        n_bkg=(1 - signal_frac_guess) * n_tot,
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
        (0, n_tot),  # N sig
        (0, n_tot),  # N bkg
        (144.0, 147.0),  # Centre
        (0.1, 1.0),  # width L
        (0.1, 1.0),  # width R
        (0.0, 2.0),  # alpha L
        (0.0, 2.0),  # alpha R
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

    rs_a, rs_b = pdfs.background_defaults("cf")
    ws_a, ws_b = pdfs.background_defaults("dcs")

    chi2 = pdfs.SimultaneousBinnedChi2(rs_counts, ws_counts, bins, rs_errors, ws_errors)

    n_rs = np.sum(rs_counts)
    n_ws = np.sum(ws_counts)
    rs_frac_guess, ws_frac_guess = 0.95, 0.05
    m = Minuit(
        chi2,
        rs_n_sig=n_rs * rs_frac_guess,
        rs_n_bkg=n_rs * (1 - rs_frac_guess),
        ws_n_sig=n_ws * ws_frac_guess,
        ws_n_bkg=n_ws * (1 - ws_frac_guess),
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        rs_a=rs_a,
        rs_b=rs_b,
        ws_a=ws_a,
        ws_b=ws_b,
    )
    m.limits["rs_n_sig"] = (0.0, n_rs)
    m.limits["ws_n_sig"] = (0.0, n_ws)
    m.limits["rs_n_bkg"] = (0.0, n_rs)
    m.limits["ws_n_bkg"] = (0.0, n_ws)
    m.limits["centre"] = (144.0, 147.0)
    m.limits["width_l"] = (0.1, 1.0)
    m.limits["width_r"] = (0.1, 1.0)
    m.limits["alpha_l"] = (0.0, 2.0)
    m.limits["alpha_r"] = (0.0, 2.0)

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

    fit_yields = fitter.values[:2]
    fit_errs = fitter.errors[:2]

    if path:
        if rs_errors is None:
            rs_errors = np.sqrt(rs_counts)
        if ws_errors is None:
            ws_errors = np.sqrt(ws_counts)

        fig, _ = plotting.simul_fits(
            rs_counts, rs_errors, ws_counts, ws_errors, bins, fitter.values
        )
        fig.suptitle(f"{fitter.valid=}")
        fig.tight_layout()

        print(f"Saving {path}")
        plt.savefig(path)
        plt.close(fig)

    return fit_yields, fit_errs


def alt_bkg_fit(
    counts: np.ndarray,
    bins: np.ndarray,
    sign: str,
    time_bin: int,
    signal_frac_guess: float,
    *,
    errors: np.ndarray = None,
    bdt_cut: bool = False,
    efficiency: bool = False,
) -> Minuit:
    """
    Perform a binned fit with the alternate bkg, return the fitter

    :param counts: array of D* - D0 mass differences
    :param bins: delta M binning used for the fit
    :param sign: either "cf" or "dcs"
    :param time_bin: which time bin we're performing the fit in; this determines the value of beta
    :param signal_frac_guess: initial guess at the signal fraction
    :param errors: optional errors. If not provided Poisson errors assumed
    :param bdt_cut: whether to model the background after the BDT cut
    :param efficiency: whether to model the background after the efficiency correction

    :returns: fitter after performing the fit

    """
    assert sign in {"cf", "dcs"}
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a_0, a_1, a_2 = 0.0, 0.0, 0.0

    # Get the bkg pdf from a pickle dump
    bkg_pdf = bkg.pdf(len(counts), sign, bdt_cut=bdt_cut, efficiency=efficiency)

    chi2 = pdfs.AltBkgBinnedChi2(bkg_pdf, counts, bins, errors)

    n_tot = np.sum(counts)
    m = Minuit(
        chi2,
        n_sig=signal_frac_guess * n_tot,
        n_bkg=(1 - signal_frac_guess) * n_tot,
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        a_0=a_0,
        a_1=a_1,
        a_2=a_2,
    )
    m.limits = (
        (0, n_tot),  # N sig
        (0, n_tot),  # N bkg
        (144.0, 147.0),  # Centre
        (0.1, 1.0),  # width L
        (0.1, 1.0),  # width R
        (0.0, 2.0),  # alpha L
        (0.0, 2.0),  # alpha R
        (None, None),  # Beta (fixed below)
        (None, None),  # Background a0
        (None, None),  # Background a1
        (None, None),  # Background a2
    )

    m.fixed["beta"] = True

    m.migrad()

    return m


def alt_simultaneous_fit(
    rs_counts: np.ndarray,
    ws_counts: np.ndarray,
    bins: np.ndarray,
    time_bin: int,
    rs_errors: np.ndarray = None,
    ws_errors: np.ndarray = None,
    *,
    bdt_cut: bool = False,
    efficiency: bool = False,
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
    assert len(rs_counts) == len(ws_counts)
    assert len(bins) - 1 == len(ws_counts)

    centre, width_l, alpha_l, beta = pdfs.signal_defaults(time_bin)
    width_r, alpha_r = width_l, alpha_l

    a_0, a_1, a_2 = 0.0, 0.0, 0.0
    # Get the bkg pdf from a pickle dump
    cf_bkg = bkg.pdf(len(rs_counts), "cf", bdt_cut=bdt_cut, efficiency=efficiency)
    dcs_bkg = bkg.pdf(len(rs_counts), "dcs", bdt_cut=bdt_cut, efficiency=efficiency)

    chi2 = pdfs.SimulAltBkg(
        cf_bkg, dcs_bkg, rs_counts, ws_counts, bins, rs_errors, ws_errors
    )

    n_rs = np.sum(rs_counts)
    n_ws = np.sum(ws_counts)
    rs_frac_guess, ws_frac_guess = 0.95, 0.05
    m = Minuit(
        chi2,
        rs_n_sig=n_rs * rs_frac_guess,
        rs_n_bkg=n_rs * (1 - rs_frac_guess),
        ws_n_sig=n_ws * ws_frac_guess,
        ws_n_bkg=n_ws * (1 - ws_frac_guess),
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        rs_a_0=a_0,
        rs_a_1=a_1,
        rs_a_2=a_2,
        ws_a_0=a_0,
        ws_a_1=a_1,
        ws_a_2=a_2,
    )
    m.limits["rs_n_sig"] = (0, n_rs)
    m.limits["ws_n_sig"] = (0, n_ws)
    m.limits["rs_n_bkg"] = (0.0, n_rs)
    m.limits["ws_n_bkg"] = (0.0, n_ws)
    m.limits["centre"] = (144.0, 147.0)
    m.limits["width_l"] = (0.1, 1.0)
    m.limits["width_r"] = (0.1, 1.0)
    m.limits["alpha_l"] = (0.0, 2.0)
    m.limits["alpha_r"] = (0.0, 2.0)

    m.fixed["beta"] = True

    m.migrad(ncall=5000)

    if m.valid:
        m.minos()
    else:
        print(m)

    return m
