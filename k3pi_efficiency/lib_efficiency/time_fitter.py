"""
Fitter for Monte Carlo decay times (i.e. after the efficiency
has been applied)

Everything is in units of D decay time

"""
from typing import Tuple
import numpy as np
from scipy.integrate import quad
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL


def _pdf_no_min_t(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, k: float
):
    """
    PDF without accounting for the minimum time

    """
    delta_t = t - t0
    numerator = (delta_t ** (n + m)) * np.exp(-k * t)

    denominator = (1 + (a * delta_t) ** n) * (1 + (b * delta_t) ** m)

    return numerator / denominator


def pdf(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, k: float
) -> np.ndarray:
    """
    PDF model for decay time distribution

    """
    # Events above the min time
    above_min = t > t0

    # Init to 0, so that events we dont set are 0
    retval = np.zeros_like(t)

    retval[above_min] = _pdf_no_min_t(t[above_min], t0, n, m, a, b, k)

    return retval


def integral(t0: float, n: float, m: float, a: float, b: float, k: float) -> float:
    """
    Integral of the PDF

    """
    return quad(_pdf_no_min_t, t0, np.inf, args=(t0, n, m, a, b, k))[0]


def normalised_pdf(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, k: float
) -> Tuple[int, np.ndarray]:
    """
    Normalised PDF model for decay time distribution

    Returns length of the t array and the pdf values

    """
    return len(t), pdf(t, t0, n, m, a, b, k) / integral(t0, n, m, a, b, k)


def fit(
    times: np.ndarray, initial_guess: Tuple[float, float, float, float, float, float]
) -> Minuit:
    """
    Perform a fit to some decay times

    """
    nll = ExtendedUnbinnedNLL(times, normalised_pdf, verbose=0)

    m = Minuit(nll, *initial_guess)

    m.limits["n"] = (0.5, 3.0)
    m.limits["m"] = (0.1, 3.0)
    m.limits["a"] = (0.1, 2.0)
    m.limits["b"] = (0.5, 3.0)
    m.limits["k"] = (0.9, 1.2)

    m.fixed["t0"] = True

    m.migrad()

    return m
