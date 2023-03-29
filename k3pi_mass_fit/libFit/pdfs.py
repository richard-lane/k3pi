"""
PDFs, CDF, integrals etc. for mass fit signal and background shapes

"""
from typing import Tuple, Callable
import numpy as np
from scipy.integrate import quad
from iminuit import Minuit
from iminuit.util import make_func_code


class ZeroCountsError(Exception):
    """So we can catch this specific error"""

    def __init__(self, msg: str):
        """If there are 0s in the counts"""
        super().__init__(msg)


def domain() -> Tuple[float, float]:
    """Edges of the delta M range"""
    return 139.57, 152.0


def reduced_domain() -> Tuple[float, float]:
    """Edges of the delta M range for the smaller region fit"""
    return 140.0, domain()[1]


def background(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    PDF model for background distribution in delta M (D* - D0 mass)

    :param x: independent variable, mass difference
    :param a: shape parameter
    :param b: shape parameter
    :returns: PDF at x evaluated with the given parameters

    """
    tmp = np.sqrt(x - domain()[0])
    return tmp + a * tmp**3 + b * tmp**5


def _bkg_integral_dispatcher(x: float, a: float, b: float) -> float:
    """Integral of the bkg pdf up to x"""
    tmp = np.sqrt(x - domain()[0])
    return 2 * tmp**3 / 3 + 2 * a * tmp**5 / 5 + 2 * b * tmp**7 / 7


def normalised_bkg(x: np.ndarray, a: float, b: float, pdf_domain: Tuple) -> np.ndarray:
    """
    Normalised bkg PDF; normalised in pdf_domain

    """
    low, high = pdf_domain
    area = _bkg_integral_dispatcher(high, a, b) - _bkg_integral_dispatcher(low, a, b)

    return background(x, a, b) / area


def signal_base(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    """Signal model for delta M distribution"""
    diff_sq = (x - centre) ** 2

    numerator = diff_sq * (1 + beta * diff_sq)
    denominator = 2 * width**2 + alpha * diff_sq

    return np.exp(-numerator / denominator)


def signal(
    x: np.ndarray,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
) -> np.ndarray:
    """
    Signal model for delta M distribution, with different left/right widths + alpha

    """
    retval = np.empty(len(x))

    lower = x < centre
    retval[lower] = signal_base(x[lower], centre, width_l, alpha_l, beta)

    higher = ~lower
    retval[higher] = signal_base(x[higher], centre, width_r, alpha_r, beta)

    return retval


def normalised_signal(
    x: np.ndarray,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
    pdf_domain: Tuple,
):
    """Normalised signal PDF"""
    left_args = (centre, width_l, alpha_l, beta)
    right_args = (centre, width_r, alpha_r, beta)
    low, high = pdf_domain

    eps = 1e-10
    quad_args = {"epsabs": eps, "epsrel": eps, "points": [centre]}
    area = (
        quad(signal_base, low, centre, args=left_args, **quad_args)[0]
        + quad(signal_base, centre, high, args=right_args, **quad_args)[0]
    )

    return signal(x, centre, width_l, width_r, alpha_l, alpha_r, beta) / area


def model(
    x: np.ndarray,
    n_sig: float,
    n_bkg: float,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
    a: float,
    b: float,
    pdf_domain: Tuple,
) -> np.ndarray:
    """
    Fit model including the right number of signal and background events

    """
    return n_sig * normalised_signal(
        x, centre, width_l, width_r, alpha_l, alpha_r, beta, pdf_domain
    ) + n_bkg * normalised_bkg(x, a, b, pdf_domain)


class BinnedChi2:
    """
    Cost function for binned mass fit

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        counts: np.ndarray,
        bins: np.ndarray,
        pdf_domain: Tuple[float, float],
        error: np.ndarray = None,
    ):
        """
        Set things we need for the fit

        If error not provided, Poisson errors assumed

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "n_sig",
                "n_bkg",
                "centre",
                "width_l",
                "width_r",
                "alpha_l",
                "alpha_r",
                "beta",
                "a",
                "b",
            ]
        )

        if error is None:
            error = np.sqrt(counts)

        # Check for zeros since this breaks stuff
        if np.any(counts == 0.0) or np.any(error == 0.0):
            raise ZeroCountsError(f"{counts=}\t{error=}")

        self._counts, self._error = counts, error
        self._bins = bins
        self._domain = pdf_domain

        # For evaluating the integral of the PDF over the bins
        # This gives us n_pts_per_bin pts between bin edges
        self._n_pts_per_bin = 51

        # This is slow but only gets called once
        self._integral_pts = np.unique(
            np.concatenate(
                [
                    np.linspace(a, b, self._n_pts_per_bin + 2)
                    for a, b in zip(self._bins[:-1], self._bins[1:])
                ]
            )
        )
        self._integral_h = self._integral_pts[1:] - self._integral_pts[:-1]

    def __call__(
        self,
        n_sig: float,
        n_bkg: float,
        centre: float,
        width_l: float,
        width_r: float,
        alpha_l: float,
        alpha_r: float,
        beta: float,
        a: float,
        b: float,
    ) -> float:
        """
        Objective function

        """
        # In each bin [a, b] the predicted number
        # is int_a^b f(x) dx where f(x) is our model fcn
        # Find this integral by evaluating the function at lots of points
        integral_vals = model(
            self._integral_pts,
            n_sig,
            n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            a,
            b,
            self._domain,
        )

        areas = 0.5 * (integral_vals[1:] + integral_vals[:-1]) * self._integral_h
        predicted = np.sum(
            areas.reshape(len(self._bins) - 1, self._n_pts_per_bin + 1), axis=1
        )

        return np.sum((self._counts - predicted) ** 2 / self._error**2)


class SimultaneousBinnedChi2:
    """
    Cost function for binned mass fit,
    simultaneous RS and WS fit

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        rs_counts: np.ndarray,
        ws_counts: np.ndarray,
        bins: np.ndarray,
        pdf_domain: Tuple[float, float],
        rs_error: np.ndarray = None,
        ws_error: np.ndarray = None,
    ):
        """
        Set things we need for the fit

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "rs_n_sig",
                "rs_n_bkg",
                "ws_n_sig",
                "ws_n_bkg",
                "centre",
                "width_l",
                "width_r",
                "alpha_l",
                "alpha_r",
                "beta",
                "rs_a",
                "rs_b",
                "ws_a",
                "ws_b",
            ]
        )
        self.rs_chi2 = BinnedChi2(rs_counts, bins, pdf_domain, rs_error)
        self.ws_chi2 = BinnedChi2(ws_counts, bins, pdf_domain, ws_error)

    def __call__(
        self,
        rs_n_sig: float,
        rs_n_bkg: float,
        ws_n_sig: float,
        ws_n_bkg: float,
        centre: float,
        width_l: float,
        width_r: float,
        alpha_l: float,
        alpha_r: float,
        beta: float,
        rs_a: float,
        rs_b: float,
        ws_a: float,
        ws_b: float,
    ) -> float:
        """
        Objective function

        """
        return self.rs_chi2(
            rs_n_sig,
            rs_n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            rs_a,
            rs_b,
        ) + self.ws_chi2(
            ws_n_sig,
            ws_n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            ws_a,
            ws_b,
        )


def estimated_bkg(
    x: float,
    pdf: Callable,
    pdf_domain: Tuple[float, float],
    a_0: float,
    a_1: float,
    a_2: float,
) -> float:
    """
    Background model from estimated bkg

    :param x: points
    :param pdf: estimated bkg fcn - must be normalised
    :params a_: polynomial coeffs

    """
    # Find the integral of the PDF so we can scale
    a, b = pdf_domain
    width = b - a
    integral = 1 + a_0 * width + a_1 * width**2 / 2 + a_2 * width**3 / 3

    return (pdf(x) + a_0 + a_1 * (x - a) + a_2 * (x - a) ** 2) / integral


def model_alt_bkg(
    x: np.ndarray,
    n_sig: float,
    n_bkg: float,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
    bkg_pdf: Callable,
    pdf_domain: Tuple,
    a_0: float,
    a_1: float,
    a_2: float,
) -> np.ndarray:
    """
    Fit model including the right number of signal and background events
    with the alternate background model

    """
    return n_sig * normalised_signal(
        x,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        pdf_domain,
    ) + n_bkg * estimated_bkg(x, bkg_pdf, pdf_domain, a_0, a_1, a_2)


class AltBkgBinnedChi2:
    """
    Cost function for binned mass fit

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        bkg_pdf: Callable,
        counts: np.ndarray,
        bins: np.ndarray,
        error: np.ndarray = None,
    ):
        """
        Set things we need for the fit

        If error not provided, Poisson errors assumed

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "n_sig",
                "n_bkg",
                "centre",
                "width_l",
                "width_r",
                "alpha_l",
                "alpha_r",
                "beta",
                "a_0",
                "a_1",
                "a_2",
            ]
        )

        if error is None:
            error = np.sqrt(counts)

        self.counts, self.error = counts, error
        self.centres = (bins[1:] + bins[:-1]) / 2
        self.widths = bins[1:] - bins[:-1]
        self.pdf = bkg_pdf
        self.domain = bins[0], bins[-1]

    def __call__(
        self,
        n_sig: float,
        n_bkg: float,
        centre: float,
        width_l: float,
        width_r: float,
        alpha_l: float,
        alpha_r: float,
        beta: float,
        a_0: float,
        a_1: float,
        a_2: float,
    ) -> float:
        """
        Objective function

        """
        # Can't use stats.areas as that requires us to
        # evaluate the PDF at each bin edge, which we can't
        # do for the alt bkg (since it overflows)
        # Instead evaluate at centres
        predicted = self.widths * model_alt_bkg(
            self.centres,
            n_sig,
            n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            self.pdf,
            self.domain,
            a_0,
            a_1,
            a_2,
        )

        return np.sum((self.counts - predicted) ** 2 / self.error**2)


class SimulAltBkg:
    """
    Cost function for binned mass fit,
    simultaneous RS and WS fit with alt bkg model

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        rs_bkg_pdf: np.ndarray,
        ws_bkg_pdf: np.ndarray,
        rs_counts: np.ndarray,
        ws_counts: np.ndarray,
        bins: np.ndarray,
        rs_error: np.ndarray = None,
        ws_error: np.ndarray = None,
    ):
        """
        Set things we need for the fit

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "rs_n_sig",
                "rs_n_bkg",
                "ws_n_sig",
                "ws_n_bkg",
                "centre",
                "width_l",
                "width_r",
                "alpha_l",
                "alpha_r",
                "beta",
                "rs_a_0",
                "rs_a_1",
                "rs_a_2",
                "ws_a_0",
                "ws_a_1",
                "ws_a_2",
            ]
        )
        self.rs_chi2 = AltBkgBinnedChi2(rs_bkg_pdf, rs_counts, bins, rs_error)
        self.ws_chi2 = AltBkgBinnedChi2(ws_bkg_pdf, ws_counts, bins, ws_error)

    def __call__(
        self,
        rs_n_sig: float,
        rs_n_bkg: float,
        ws_n_sig: float,
        ws_n_bkg: float,
        centre: float,
        width_l: float,
        width_r: float,
        alpha_l: float,
        alpha_r: float,
        beta: float,
        rs_a_0: float,
        rs_a_1: float,
        rs_a_2: float,
        ws_a_0: float,
        ws_a_1: float,
        ws_a_2: float,
    ) -> float:
        """
        Objective function

        """
        return self.rs_chi2(
            rs_n_sig,
            rs_n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            rs_a_0,
            rs_a_1,
            rs_a_2,
        ) + self.ws_chi2(
            ws_n_sig,
            ws_n_bkg,
            centre,
            width_l,
            width_r,
            alpha_l,
            alpha_r,
            beta,
            ws_a_0,
            ws_a_1,
            ws_a_2,
        )
