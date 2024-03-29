"""
Function for doing the mass fit

"""
from typing import Tuple, Callable
import numpy as np
from iminuit import Minuit

from . import pdfs


def binned_fit(
    counts: np.ndarray,
    bins: np.ndarray,
    initial_guess: Tuple,
    pdf_domain: Tuple[float, float],
    *,
    errors: np.ndarray = None,
) -> Minuit:
    """
    Perform a binned fit, return the fitter

    :param counts: array of D* - D0 mass differences
    :param bins: delta M binning used for the fit
    :param initial_guess: initial guess at parameters,
                          N_sig, N_bkg, centre, width_l, width_r, alpha_l
                          alpha_r, a, b
    :param pdf_domain: values to define the PDFs over for the fit
    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    assert len(pdf_domain) == 2
    assert bins[0] == pdf_domain[0]
    assert bins[-1] == pdf_domain[1]

    (
        n_sig,
        n_bkg,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        bkg_a,
        bkg_b,
    ) = initial_guess

    chi2 = pdfs.BinnedChi2(counts, bins, pdf_domain, errors)

    fitter = Minuit(
        chi2,
        n_sig=n_sig,
        n_bkg=n_bkg,
        centre=centre,
        width_l=width_l,
        width_r=width_r,
        alpha_l=alpha_l,
        alpha_r=alpha_r,
        beta=beta,
        a=bkg_a,
        b=bkg_b,
    )

    n_tot = np.sum(counts)
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

    if fitter.valid:
        fitter.hesse()

    return fitter


def binned_simultaneous_fit(
    rs_counts: np.ndarray,
    ws_counts: np.ndarray,
    bins: np.ndarray,
    initial_guess: Tuple,
    pdf_domain: Tuple,
    *,
    rs_errors: np.ndarray = None,
    ws_errors: np.ndarray = None,
) -> Minuit:
    """
    Perform the fit, return the fitter

    :param rs_counts: D* - D0 mass difference counts in the bins
    :param ws_counts: D* - D0 mass difference counts in the bins
    :param bins: delta M binning
    :param initial_guess: inital guess at the parameters
    :param rs_errors: optional bin errors. Poisson errors assumed otherwise
    :param ws_errors: optional bin errors. Poisson errors assumed otherwise

    :returns: fitter after performing the fit

    """
    assert bins[0] == pdf_domain[0]
    assert bins[-1] == pdf_domain[1]

    (
        n_rs_sig,
        n_rs_bkg,
        n_ws_sig,
        n_ws_bkg,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        rs_a,
        rs_b,
        ws_a,
        ws_b,
    ) = initial_guess

    chi2 = pdfs.SimultaneousBinnedChi2(
        rs_counts, ws_counts, bins, pdf_domain, rs_errors, ws_errors
    )

    n_rs = np.sum(rs_counts)
    n_ws = np.sum(ws_counts)
    fitter = Minuit(
        chi2,
        rs_n_sig=n_rs_sig,
        rs_n_bkg=n_rs_bkg,
        ws_n_sig=n_ws_sig,
        ws_n_bkg=n_ws_bkg,
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
    fitter.limits["rs_n_sig"] = (0.0, n_rs)
    fitter.limits["ws_n_sig"] = (0.0, n_ws)
    fitter.limits["rs_n_bkg"] = (0.0, n_rs)
    fitter.limits["ws_n_bkg"] = (0.0, n_ws)
    fitter.limits["centre"] = (144.0, 147.0)
    fitter.limits["width_l"] = (0.1, 1.0)
    fitter.limits["width_r"] = (0.1, 1.0)
    fitter.limits["alpha_l"] = (0.0, 2.0)
    fitter.limits["alpha_r"] = (0.0, 2.0)

    fitter.fixed["beta"] = True

    fitter.migrad(ncall=5000)

    fitter.hesse()

    return fitter


def alt_bkg_fit(
    counts: np.ndarray,
    bins: np.ndarray,
    initial_guess: Tuple,
    bkg_pdf: Callable,
    *,
    errors: np.ndarray = None,
) -> Minuit:
    """
    Perform a binned fit with the alternate bkg, return the fitter

    :param counts: array of D* - D0 mass differences
    :param bins: delta M binning used for the fit
    :param bkg_pdf: bkg PDF, normalised over the bins
    :param initial_guess:  initial guess at the parameters

    :param errors: optional errors. If not provided Poisson errors assumed

    :returns: fitter after performing the fit

    """
    assert len(counts) == len(bins) - 1
    if (errors is not None) and (len(errors) != len(counts)):
        raise ValueError(f"{len(errors)=}\t{len(counts)=}")

    chi2 = pdfs.AltBkgBinnedChi2(bkg_pdf, counts, bins, errors)

    (
        n_sig,
        n_bkg,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        a_0,
        a_1,
        a_2,
    ) = initial_guess

    fitter = Minuit(
        chi2,
        n_sig=n_sig,
        n_bkg=n_bkg,
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

    n_tot = np.sum(counts)
    fitter.limits = (
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

    fitter.fixed["beta"] = True

    fitter.migrad()

    return fitter


def alt_simultaneous_fit(
    rs_counts: np.ndarray,
    ws_counts: np.ndarray,
    bins: np.ndarray,
    initial_guess: Tuple,
    rs_bkg_pdf: Callable,
    ws_bkg_pdf: Callable,
    *,
    rs_errors: np.ndarray = None,
    ws_errors: np.ndarray = None,
) -> Minuit:
    """
    Perform the fit, return the fitter

    :param rs_counts: D* - D0 mass difference counts in the bins
    :param ws_counts: D* - D0 mass difference counts in the bins
    :param bins: delta M binning
    :param initial_guess: initial guess at params
    :param rs_bkg_pdf: bkg pdf, normalised in fit region
    :param ws_bkg_pdf: bkg pdf, normalised in fit region

    :param rs_errors: optional bin errors. Poisson errors assumed otherwise
    :param ws_errors: optional bin errors. Poisson errors assumed otherwise

    :returns: fitter after performing the fit

    """
    assert len(rs_counts) == len(ws_counts)
    assert len(bins) - 1 == len(ws_counts)

    (
        rs_n_sig,
        rs_n_bkg,
        ws_n_sig,
        ws_n_bkg,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        rs_a0,
        rs_a1,
        rs_a2,
        ws_a0,
        ws_a1,
        ws_a2,
    ) = initial_guess

    # Get the bkg pdfs from a pickle dump

    chi2 = pdfs.SimulAltBkg(
        rs_bkg_pdf, ws_bkg_pdf, rs_counts, ws_counts, bins, rs_errors, ws_errors
    )

    # Repeat until fit converges, unless it doesn't
    done = False
    n_iter = 0
    while not done:
        fitter = Minuit(
            chi2,
            rs_n_sig=rs_n_sig,
            rs_n_bkg=rs_n_bkg,
            ws_n_sig=ws_n_sig,
            ws_n_bkg=ws_n_bkg,
            centre=centre,
            width_l=width_l,
            width_r=width_r,
            alpha_l=alpha_l,
            alpha_r=alpha_r,
            beta=beta,
            rs_a_0=rs_a0,
            rs_a_1=rs_a1,
            rs_a_2=rs_a2,
            ws_a_0=ws_a0,
            ws_a_1=ws_a1,
            ws_a_2=ws_a2,
        )

        n_rs = np.sum(rs_counts)
        n_ws = np.sum(ws_counts)
        fitter.limits["rs_n_sig"] = (0, n_rs)
        fitter.limits["ws_n_sig"] = (0, n_ws)
        fitter.limits["rs_n_bkg"] = (0.0, n_rs)
        fitter.limits["ws_n_bkg"] = (0.0, n_ws)
        fitter.limits["centre"] = (144.0, 147.0)
        fitter.limits["width_l"] = (0.1, 1.0)
        fitter.limits["width_r"] = (0.1, 1.0)
        fitter.limits["alpha_l"] = (0.0, 2.0)
        fitter.limits["alpha_r"] = (0.0, 2.0)

        fitter.fixed["beta"] = True

        fitter.migrad(ncall=8000, iterate=10)

        done = fitter.valid

        if not done:
            print(f"ALT BKG:")
            print(fitter)

        n_iter += 1
        if n_iter > 5:
            break

    return fitter
