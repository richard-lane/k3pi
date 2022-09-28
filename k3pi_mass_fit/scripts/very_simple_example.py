"""
Very simple fit example with unbinned/binned fits

"""
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from typing import Callable
from scipy.integrate import quad
from iminuit.cost import ExtendedUnbinnedNLL, ExtendedBinnedNLL


def pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    A straight line

    """
    return a + b * x


def pdf_with_integral(x, a, b, domain, n):
    return n * quad(pdf, *domain, args=(a, b))[0], pdf(x, a, b)


def _gen(
    rng: np.random.Generator,
    pdf: Callable[[float], float],
    pdf_max: float,
    pdf_domain: tuple,
) -> np.ndarray:
    """
    Generate samples from a pdf

    """
    n_gen = 100000
    x = pdf_domain[0] + (pdf_domain[1] - pdf_domain[0]) * rng.random(n_gen)

    y = pdf_max * rng.random(n_gen)

    f_eval = pdf(x)

    keep = y < f_eval

    return x, y, keep


def _max(pdf: Callable, domain: tuple) -> float:
    """
    Find the maximum value of a function of 1 dimension

    Then multiply it by 1.1 just to be safe

    """
    return 1.1 * np.max(pdf(np.linspace(*domain, 100)))


def main():
    # Generate points from the pdf
    domain = (0, 1)
    true_a, true_b = 1.0, 0.5
    true_pdf = lambda x: pdf(x, true_a, true_b)
    x, y, keep = _gen(
        np.random.default_rng(seed=0), true_pdf, _max(true_pdf, domain), domain
    )

    points = x[keep]
    n = len(points)
    print(n)

    # Fit them back
    c = ExtendedUnbinnedNLL(
        points, lambda x, a, b: pdf_with_integral(x, a, b, domain, n), verbose=1
    )

    m = Minuit(c, a=true_a, b=true_b)
    m.migrad()

    for p in m.params:
        print(p)

    fitted_pdf = lambda x: pdf(x, *m.values)

    # Plot
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(points, y[keep], color="r", s=0.1)
    ax[0].scatter(x[~keep], y[~keep], color="b", s=0.1)

    pts = np.linspace(*domain, 200)
    vals = pdf(pts, true_a, true_b)
    ax[0].plot(pts, vals, "k")

    ax[1].set_title(f"m={m.values[1]:.2f}  c={m.values[0]:.2f}")
    fig.suptitle(f"{n=}")
    ax[1].hist(points, bins=100, density=True)
    ax[1].plot(pts, fitted_pdf(pts), "r", label="fit")

    ax[1].plot(pts, vals / quad(true_pdf, *domain)[0], "k--", label="true")
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
