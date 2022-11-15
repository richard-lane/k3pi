"""
PDFs, CDF, integrals etc. for mass fit signal and background shapes

"""
from typing import Tuple
import numpy as np
from scipy.integrate import quad
from iminuit import Minuit
from iminuit.util import make_func_code


def domain() -> Tuple[float, float]:
    """Edges of the delta M range"""
    return 139.57, 152.0


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
    return 2 / 3 * tmp**3 + 2 * a / 5 * tmp**5 + 2 * b / 7 * tmp**7


def _bkg_integral(low: float, high: float, a: float, b: float) -> float:
    """Normalised integral"""
    return (
        _bkg_integral_dispatcher(high, a, b) - _bkg_integral_dispatcher(low, a, b)
    ) / _bkg_integral_dispatcher(domain()[1], a, b)


def normalised_bkg(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Normalised bkg PDF"""
    # The integral across the whole domain is equal to the integral evaluated at the
    # high limit
    return background(x, a, b) / _bkg_integral_dispatcher(domain()[1], a, b)


def signal_base(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    """Signal model for delta M distribution"""
    diff_sq = (x - centre) ** 2

    numerator = diff_sq * (1 + beta * diff_sq)
    denominator = 2 * width**2 + alpha * diff_sq

    return np.exp(-numerator / denominator)


def _signal_base_integral(x_domain: Tuple[float, float], args: Tuple) -> float:
    """Non-normalised integral of the signal model"""
    return quad(signal_base, *x_domain, args=args)[0]


def normalised_sig_base(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Signal shape, normalised

    """
    args = (centre, width, alpha, beta)
    area = _signal_base_integral(domain(), args)

    return signal_base(x, *args) / area


def _norm_sig_base_integral(x_domain: Tuple[float, float], args: Tuple) -> float:
    """Normalised integral of the signal model"""
    return quad(normalised_sig_base, *x_domain, args=args)[0]


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
    Signal model for delta M distribution, with different left/right widths

    """
    retval = np.ndarray(len(x))

    lower = x < centre
    retval[lower] = signal_base(x[lower], centre, width_l, alpha_l, beta)

    higher = ~lower
    retval[higher] = signal_base(x[higher], centre, width_r, alpha_r, beta)

    return retval


def _signal_integral(
    x_domain: Tuple[float, float],
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
) -> float:
    """Integral of the normalised signal PDF between two points"""
    assert x_domain[0] <= x_domain[1]
    left_args = (centre, width_l, alpha_l, beta)
    right_args = (centre, width_r, alpha_r, beta)

    # Entire range above the centre
    if x_domain[0] > centre:
        return _norm_sig_base_integral(x_domain, args=right_args)

    # Entire range below the centre
    if x_domain[1] < centre:
        return _norm_sig_base_integral(x_domain, args=left_args)

    # Range crosses the centre
    return _norm_sig_base_integral(
        (x_domain[0], centre), args=left_args
    ) + _norm_sig_base_integral((centre, x_domain[1]), args=left_args)


def normalised_signal(
    x: np.ndarray,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
):
    """Normalised signal PDF"""
    left_args = (centre, width_l, alpha_l, beta)
    right_args = (centre, width_r, alpha_r, beta)
    low, high = domain()

    area = (
        quad(signal_base, low, centre, args=left_args)[0]
        + quad(signal_base, centre, high, args=right_args)[0]
    )

    return signal(x, centre, width_l, width_r, alpha_l, alpha_r, beta) / area


def fractional_pdf(
    x: np.ndarray,
    signal_fraction: float,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
    a: float,
    b: float,
) -> np.ndarray:
    """
    returns n_evts, n_sig * normalised sig pdf + n_bkg * normalised bkg pdf

    """
    return signal_fraction * normalised_signal(
        x,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
    ) + (1 - signal_fraction) * normalised_bkg(x, a, b)


def pdf(
    x: np.ndarray,
    signal_fraction: float,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
    a: float,
    b: float,
) -> Tuple[int, np.ndarray]:
    """
    returns n_evts, n_sig * normalised sig pdf + n_bkg * normalised bkg pdf

    """
    n = len(x)
    return n, n * fractional_pdf(
        x, signal_fraction, centre, width_l, width_r, alpha_l, alpha_r, beta, a, b
    )


class BinnedChi2:
    """
    Cost function for binned mass fit

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
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
                "signal_fraction",
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

        self.counts, self.error = counts, error
        self.total = np.sum(self.counts)

        self.centres = (bins[1:] + bins[:-1]) / 2
        self.widths = (bins[1:] - bins[:-1]) / 2

    def __call__(
        self,
        signal_fraction: float,
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
        return np.sum(
            (
                self.counts
                - (
                    self.total
                    * self.widths
                    * fractional_pdf(
                        self.centres,
                        signal_fraction,
                        centre,
                        width_l,
                        width_r,
                        alpha_l,
                        alpha_r,
                        beta,
                        a,
                        b,
                    )
                )
            )
            ** 2
            / self.error**2
        )


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
        rs_error: np.ndarray = None,
        ws_error: np.ndarray = None,
    ):
        """
        Set things we need for the fit

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(
            [
                "rs_signal_fraction",
                "ws_signal_fraction",
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
        self.rs_chi2 = BinnedChi2(rs_counts, bins, rs_error)
        self.ws_chi2 = BinnedChi2(ws_counts, bins, ws_error)

    def __call__(
        self,
        rs_signal_fraction: float,
        ws_signal_fraction: float,
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
        return self.rs_chi2(
            rs_signal_fraction, centre, width_l, width_r, alpha_l, alpha_r, beta, a, b
        ) + self.ws_chi2(
            ws_signal_fraction, centre, width_l, width_r, alpha_l, alpha_r, beta, a, b
        )


def default_centre() -> float:
    """Initial guess for the centre of the signal peak"""
    return 146.0


def signal_defaults(time_bin: int) -> Tuple[float, float, float, float]:
    """
    the default values of the signal params (centre, width, alpha, beta)

    """
    return default_centre(), 0.2, 0.18, 0.0019 * time_bin + 0.0198


def background_defaults(sign: str) -> Tuple[float, float]:
    """
    default values for background parameters (a, b), for either "WS" or "WS"

    """
    assert sign in {"RS", "WS"}
    if sign == "WS":
        return 0.06, -0.00645

    return 0.004, -0.001


def defaults(
    sign: str, time_bin: int
) -> Tuple[float, float, float, float, float, float]:
    """
    default centre, width, alpha, beta, a, b

    """
    centre, width, alpha, beta = signal_defaults(time_bin)
    a, b = background_defaults(sign)

    return centre, width, alpha, beta, a, b
