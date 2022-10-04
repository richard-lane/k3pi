"""
Fit models

"""
from typing import Tuple
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from .util import ConstraintParams, MixingParams, ScanParams
from .charm_threshhold import likelihoods


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


class BaseChi2:
    """
    Base class for the chi2 cost functions

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, ratio: np.ndarray, error: np.ndarray, bins: np.ndarray):
        """
        Set common attributes for the fit - the bins, ratio etc.

        """
        self.ratio = ratio
        self.error = error
        self.bins = bins

        # Denominator (RS) integral doesn't depend on the params
        # so we only need to evaluate it once
        self.expected_rs_integral = rs_integral(bins)

    def chi2(self, expected_ratio: np.ndarray) -> float:
        """
        Evaluate chi2 from expected and actual parameters

        """
        return np.sum(((self.ratio - expected_ratio) / self.error) ** 2)


class NoConstraints(BaseChi2):
    """
    Cost function for the fitter without constraints

    """

    def __init__(self, ratio: np.ndarray, error: np.ndarray, bins: np.ndarray):
        """
        Set parameters for doing a fit without constraints

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio

        """
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["a", "b", "c"])

        super().__init__(ratio, error, bins)

    def _expected_ws_integral(self, a: float, b: float, c: float) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, a, b, c)

    def __call__(self, a: float, b: float, c: float):
        """
        Evaluate the chi2 given our parameters

        """
        return BaseChi2.chi2(
            self, self._expected_ws_integral(a, b, c) / self.expected_rs_integral
        )


class ConstrainedBase(BaseChi2):
    """
    Base class for fitters implementing a Gaussian constraint on mixing parameters x and y

    """

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
        Precompute some terms used in the

        """
        # We can pre-compute two of the terms used for the Gaussian constraint
        self._x_width, self._y_width = x_y_widths
        self._x_mean, self._y_mean = x_y_means
        self._constraint_scale = 1 / (1 - x_y_correlation**2)
        self._constraint_cross_term = (
            2 * x_y_correlation / (self._x_width * self._y_width)
        )
        super().__init__(ratio, error, bins)

    def constraint(self, x: float, y: float) -> float:
        """
        term in the chi^2 for the Gaussian constraint

        """
        delta_x = x - self._x_mean
        delta_y = y - self._y_mean

        return self._constraint_scale * (
            (delta_x / self._x_width) ** 2
            + (delta_y / self._y_width) ** 2
            - self._constraint_cross_term * delta_x * delta_y
        )


class Constraints(ConstrainedBase):
    """
    Cost function for the fitter with Gaussian constraints
    on mixing parameters x and y

    """

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
        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["r_d", "x", "y", "b"])
        super().__init__(ratio, error, bins, x_y_means, x_y_widths, x_y_correlation)

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
            / self.expected_rs_integral
        )
        chi2 = BaseChi2.chi2(self, expected_ratio)

        # Also need a term for the constraint
        constraint = super().constraint(x, y)

        return chi2 + constraint


class Scan(ConstrainedBase):
    """
    Cost function for the fitter with
    fixed Z and Gaussian constraints on x and y

    """

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
        self.re_z, self.im_z = z

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["r_d", "x", "y"])

        super().__init__(ratio, error, bins, x_y_means, x_y_widths, x_y_correlation)

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
            / self.expected_rs_integral
        )
        chi2 = BaseChi2.chi2(self, expected_ratio)

        return chi2 + super().constraint(x, y)


class CharmThreshholdScan(ConstrainedBase):
    """
    Cost function for the fitter with
    fixed Z and Gaussian constraints on x and y
    with the chi2 from the charm threshhold experiments
    (CLEO and BES-III) combined

    """

    def __init__(
        self,
        ratio: np.ndarray,
        error: np.ndarray,
        bins: np.ndarray,
        x_y_means: Tuple[float, float],
        x_y_widths: Tuple[float, float],
        x_y_correlation: float,
        z: Tuple[float, float],
        bin_number: int,
    ):
        """
        Set parameters for doing a fit with Gaussian constraint
        on x and y, and also constraints from CLEO and BES

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio
        :param x_y_means: mean for Gaussian constraint
        :param x_y_widths: widths for Gaussian constraint
        :param x_y_correlation: correlation for Gaussian constraint
        :param z: (reZ, imZ) that will be used for this fit
        :param bin_number: bin number used for the charm threshhold constraint

        """
        self.re_z, self.im_z = z
        self.bin_number = bin_number

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["r_d", "x", "y"])

        super().__init__(ratio, error, bins, x_y_means, x_y_widths, x_y_correlation)

    def _expected_ws_integral(self, params: ScanParams) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, *abc_scan(params))

    def __call__(self, r_d: float, x: float, y: float):
        """
        Evaluate the chi2 given our parameters

        """
        # If Z is outside the allowed region, the BES chi2^2 will raise some ROOT errors
        # and all bets are off. Prevent this by just returning inf if we're outside the range
        # this will cause the fit to "fail" but that's probably ok
        # TODO raise instead
        if self.re_z**2 + self.im_z**2 > 1.0:
            return np.inf

        expected_ratio = (
            self._expected_ws_integral(ScanParams(r_d, x, y, self.re_z, self.im_z))
            / self.expected_rs_integral
        )
        chi2 = BaseChi2.chi2(self, expected_ratio)

        # Also need a term for the charm threshhold
        # TODO could make this faster by preloading the CLEO
        # and BES functions from the DLLs
        threshhold_constraint = likelihoods.combined_chi2(
            self.bin_number, self.re_z, self.im_z, x, y, r_d
        )

        return chi2 + super().constraint(x, y) + threshhold_constraint
