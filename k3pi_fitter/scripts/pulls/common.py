"""
Common things for all the pull studies

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_time_fit import util, models


def xy_vals(
    gen: np.random.Generator,
    num: int,
    means: np.ndarray,
    widths: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """
    Randomly sample X and Y values from a correlated Gaussian

    """
    # Get the covariance matrix
    cov = (
        np.diag(widths)
        @ np.array([[1.0, correlation], [correlation, 1.0]])
        @ np.diag(widths)
    )

    return gen.multivariate_normal(means, cov, size=num)


def gen_rs(
    rng: np.random.Generator,
    n_gen: int,
    domain: Tuple[float, float],
) -> np.array:
    """
    Generate points from the RS distribution

    """
    # Uniform points
    uniform = rng.random(n_gen)

    low, high = domain
    exp_low, exp_high = np.exp(-low), np.exp(-high)

    return -np.log(exp_low - uniform * (exp_low - exp_high))


def n_ws(n_rs: int, domain: Tuple[float, float], params: util.MixingParams):
    """
    Number of WS evts, given some params and a number of RS

    """
    # Number of WS points is n_rs * WS integral / RS integral
    return (
        n_rs
        * models.ws_integral(np.array(domain), *params)
        / models.rs_integral(np.array(domain))
    )[0]


def gen_ws(
    rng: np.random.Generator,
    n_rs: int,
    domain: Tuple[float, float],
    params: util.MixingParams,
    plot: bool = False,
):
    """
    Generate points from the WS distribution

    Generates the right number given the number of RS and the
    decay params

    """
    num_ws = n_ws(n_rs, domain, params)
    num_ws = int(num_ws)
    retval = np.ones(num_ws) * np.nan

    def pdf(x):
        """not actually a pdf"""
        return models.no_constraints(x, *params) * np.exp(-x)

    max_pdf = pdf(fminbound(lambda x: -pdf(x), *domain))

    # Generate in chunks since we don't know how many will be accepted
    num_generated = 0
    chunk_size = 2 * num_ws
    while num_generated < num_ws:
        # Generate a chunk
        uniform = domain[0] + (domain[1] - domain[0]) * rng.random(chunk_size)
        vals = pdf(uniform)

        y = max_pdf * rng.random(chunk_size)
        keep = y < vals
        n_in_chunk = np.sum(keep)

        # Fill in with numpy array indexing
        try:
            retval[num_generated : num_generated + n_in_chunk] = uniform[keep]
            num_generated += n_in_chunk
        except ValueError:
            # Unless we go over the end, in which case just fill in the ones we need
            n_left = num_ws - np.sum(~np.isnan(retval))
            retval[num_generated:] = uniform[keep][:n_left]
            break

    if plot:
        _, ax = plt.subplots()

        pts = np.linspace(*domain, 1000)
        ax.plot(pts, pdf(pts))
        ax.scatter(uniform[keep], y[keep], c="k", marker=".")
        ax.scatter(uniform[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return retval
