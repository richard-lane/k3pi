"""
Fit models

"""
from typing import Tuple
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from .util import ConstraintParams, MixingParams, ScanParams


def abc(params: ConstraintParams) -> MixingParams:
    """
    Find a, b, c params from x, y, etc.

    """
    return MixingParams(
        params.r_d,
        params.r_d * params.b,
        0.25 * (params.x**2 + params.y**2),
    )


def scan2constraint(params: ScanParams) -> ConstraintParams:
    """
    Convert scan params to constraint params

    """
    return ConstraintParams(
        params.r_d,
        (params.x * params.im_z + params.y * params.re_z),
        params.x,
        params.y,
    )


def abc_scan(params: ScanParams) -> MixingParams:
    """
    Find a, b, c params from x, y, Z etc.

    """
    return abc(scan2constraint(params))


def no_mixing(amplitude_ratio: float) -> float:
    """
    Model for no mixing - i.e. the ratio will be a constant (amplitude_ratio^2)
    Independent of time

    :param amplitude_ratio: ratio of DCS/CF amplitudes

    :returns: amplitude_ratio ** 2

    """

    return amplitude_ratio**2


def no_constraints(times: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Model for WS/RS time ratio; allows D0 mixing but does not constrain the mixing
    parameters to their previously measured values.
    Low time/small mixing approximation

    ratio = a^2 + abt + ct^2

    :param times: times to evaluate the ratio at, in lifetimes
    :param a: amplitude ratio
    :param b: mixing parameter from interference
    :param c: mixing parameter

    :returns: ratio at each time

    """
    return a**2 + a * b * times + c * times**2


def constraints(times: np.ndarray, params: ConstraintParams) -> np.ndarray:
    """
    Model for WS/RS time ratio; allows D0 mixing,
    constraining the mixing parameters to the provided values.

    Low time/small mixing approximation

    ratio = r_d^2 + r_d * b * t + 0.25(x^2 + y^2)t^2

    :param times: times to evaluate the ratio at, in lifetimes
    :param params: mixing parameters

    :returns: ratio at each time

    """
    return no_constraints(times, *abc(params))


def scan(times: np.ndarray, params: ScanParams) -> np.ndarray:
    """
    Model for WS/RS time ratio; allows D0 mixing,
    constraining the mixing parameters to the provided values.

    Low time/small mixing approximation

    ratio = r_d^2 + r_d * (xImZ + yReZ) * t + 0.25(x^2 + y^2)t^2

    :param times: times to evaluate the ratio at, in lifetimes
    :param params: mixing parameters

    :returns: ratio at each time

    """
    return no_constraints(times, *abc_scan(params))


def rs_integral(bins: np.ndarray) -> np.ndarray:
    """
    The integral of the RS model in each bin

    (Proportional to how many RS events we expect in each bin)

    :param bins: leftmost edge of each bin, plus the rightmost edge of the last bin. In lifetimes.
    :returns: integral of e^-t in each bin

    """
    return np.exp(-bins[:-1]) - np.exp(-bins[1:])


def _ws_integral_dispatcher(
    times: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """
    Indefinite integral evaluated at each time - constant term assumed to be 0

    """
    return -np.exp(-times) * (
        (a**2) + a * b * (times + 1) + (times * (times + 2) + 2) * c
    )


def ws_integral(bins: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    The integral of the WS model in each bin

    (Proportional to how many WS events we expect in each bin -
    same constant of proportionality as the RS integral)

    :param bins: leftmost edge of each bin, plus the rightmost edge of the last bin. In lifetimes.
    :returns: integral of e^-t in each bin

    """
    return _ws_integral_dispatcher(bins[1:], a, b, c) - _ws_integral_dispatcher(
        bins[:-1], a, b, c
    )


class NoConstraints:
    """
    Cost function for the fitter without constraints

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, ratio: np.ndarray, error: np.ndarray, bins: np.ndarray):
        """
        Set parameters for doing a fit without constraints

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio

        """
        self.ratio = ratio
        self.error = error
        self.bins = bins

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["a", "b", "c"])

        # Denominator (RS) integral doesn't depend on the params
        # so we only need to evaluate it once
        self._expected_rs_integral = rs_integral(bins)

    def _expected_ws_integral(self, a: float, b: float, c: float) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, a, b, c)

    def __call__(self, a: float, b: float, c: float):
        """
        Evaluate the chi2 given our parameters

        """
        expected_ratio = (
            self._expected_ws_integral(a, b, c) / self._expected_rs_integral
        )

        return np.sum(((self.ratio - expected_ratio) / self.error) ** 2)


class Constraints:
    """
    Cost function for the fitter with Gaussian constraints
    on mixing parameters x and y

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        ratio: np.ndarray,
        error: np.ndarray,
        bins: np.ndarray,
        x_y_means: Tuple[float, float],
        x_y_widths: Tuple[float, float],
        x_y_correlation: float,
    ):
        """
        Set parameters for doing a fit without constraints

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio
        :param x_y_means: mean for Gaussian constraint
        :param x_y_widths: widths for Gaussian constraint
        :param x_y_correlation: correlation for Gaussian constraint

        """
        self.ratio = ratio
        self.error = error
        self.bins = bins

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["r_d", "x", "y", "b"])

        # Denominator (RS) integral doesn't depend on the params
        # so we only need to evaluate it once
        self._expected_rs_integral = rs_integral(bins)

        # Similarly we can pre-compute two of the terms needed
        # for the constraint term
        self._x_width, self._y_width = x_y_widths
        self._x_mean, self._y_mean = x_y_means
        self._constraint_scale = 1 / (1 - x_y_correlation**2)
        self._constraint_cross_term = (
            2 * x_y_correlation / (self._x_width * self._y_width)
        )

    def _expected_ws_integral(self, params: ScanParams) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, *abc(params))

    def __call__(self, r_d: float, x: float, y: float, b: float):
        """
        Evaluate the chi2 given our parameters

        """
        expected_ratio = (
            self._expected_ws_integral(ConstraintParams(r_d, x, y, b))
            / self._expected_rs_integral
        )

        chi2 = np.sum(((self.ratio - expected_ratio) / self.error) ** 2)

        # Also need a term for the constraint
        dx, dy = x - self._x_mean, y - self._y_mean
        constraint = self._constraint_scale * (
            (dx / self._x_width) ** 2
            + (dy / self._y_width) ** 2
            - self._constraint_cross_term * dx * dy
        )

        return chi2 + constraint


class Scan:
    """
    Cost function for the fitter with
    fixed Z and Gaussian constraints on x and y

    """

    # TODO make this share implementation with the constrained fitter
    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        ratio: np.ndarray,
        error: np.ndarray,
        bins: np.ndarray,
        x_y_means: Tuple[float, float],
        x_y_widths: Tuple[float, float],
        x_y_correlation: float,
        z: Tuple[float, float],
    ):
        """
        Set parameters for doing a fit without constraints

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio
        :param x_y_means: mean for Gaussian constraint
        :param x_y_widths: widths for Gaussian constraint
        :param x_y_correlation: correlation for Gaussian constraint
        :param z: (reZ, imZ) that will be used for this fit

        """
        self.ratio = ratio
        self.error = error
        self.bins = bins
        self.re_z, self.im_z = z

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["r_d", "x", "y"])

        # Denominator (RS) integral doesn't depend on the params
        # so we only need to evaluate it once
        self._expected_rs_integral = rs_integral(bins)

        # Similarly we can pre-compute two of the terms needed
        # for the constraint term
        self._x_width, self._y_width = x_y_widths
        self._x_mean, self._y_mean = x_y_means
        self._constraint_scale = 1 / (1 - x_y_correlation**2)
        self._constraint_cross_term = (
            2 * x_y_correlation / (self._x_width * self._y_width)
        )

    def _expected_ws_integral(self, params: ScanParams) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, *abc_scan(params))

    def __call__(self, r_d: float, x: float, y: float):
        """
        Evaluate the chi2 given our parameters

        """
        expected_ratio = (
            self._expected_ws_integral(ScanParams(r_d, x, y, self.re_z, self.im_z))
            / self._expected_rs_integral
        )

        chi2 = np.sum(((self.ratio - expected_ratio) / self.error) ** 2)

        # Also need a term for the constraint
        dx, dy = x - self._x_mean, y - self._y_mean
        constraint = self._constraint_scale * (
            (dx / self._x_width) ** 2
            + (dy / self._y_width) ** 2
            - self._constraint_cross_term * dx * dy
        )

        return chi2 + constraint
