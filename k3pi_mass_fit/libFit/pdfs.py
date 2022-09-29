import numpy as np
from typing import Tuple
from scipy.integrate import quad


def low_mass_threshhold() -> float:
    return 139.57


def domain() -> Tuple[float, float]:
    return low_mass_threshhold(), 152.0


def background(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    PDF model for background distribution in delta M (D* - D0 mass)

    :param x: independent variable, mass difference
    :param a: shape parameter
    :param b: shape parameter
    :returns: PDF at x evaluated with the given parameters

    """
    tmp = np.sqrt(x - low_mass_threshhold())
    return tmp + a * tmp**3 + b * tmp**5


def _bkg_integral_dispatcher(x: float, a: float, b: float) -> float:
    tmp = np.sqrt(x - low_mass_threshhold())
    return 2 / 3 * tmp**3 + 2 * a / 5 * tmp**5 + 2 * b / 7 * tmp**7


def _bkg_integral(a: float, b: float) -> float:
    """
    Integral of background PDF across the domain

    """
    _, high = domain()
    # The integral at x=low is 0, so we just return the integral evaluated at the upper limit
    return _bkg_integral_dispatcher(high, a, b)


def normalised_bkg(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return background(x, a, b) / _bkg_integral(a, b)


def signal_base(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Signal model for delta M distribution

    """
    diff_sq = (x - centre) ** 2

    numerator = diff_sq * (1 + beta * diff_sq)
    denominator = 2 * width**2 + alpha * diff_sq

    return np.exp(-numerator / denominator)


def normalised_sig_base(
    x: np.ndarray, centre: float, width: float, alpha: float, beta: float
) -> np.ndarray:
    args = (centre, width, alpha, beta)
    area = quad(signal_base, *domain(), args=args)[0]

    return signal_base(x, *args) / area


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


def normalised_signal(
    x: np.ndarray,
    centre: float,
    width_l: float,
    width_r: float,
    alpha_l: float,
    alpha_r: float,
    beta: float,
):
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


def default_centre() -> float:
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
