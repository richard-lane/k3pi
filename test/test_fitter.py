"""
Things where I think it's actually useful to have a unit test

"""
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_fitter"))

from lib_time_fit import models, fitter


def test_rs_integral():
    """Check we get the right thing with a really simple test"""
    assert np.allclose(models.rs_integral(np.array([0, np.inf])), np.array([1.0]))


def test_ws_integral():
    """
    Check we get the right integral by comparing with analytic integral

    """
    lims = np.array([0.0, 1.0])
    args = 1.0, 2.0, 3.0

    expected_integral = np.array([9 - 20 / np.e])
    eval_integral = models.ws_integral(lims, *args)

    assert np.allclose(eval_integral, expected_integral)


def test_mean():
    """
    Weighted mean with the same error on each point (i.e. no weights)

    """
    numbers = (1, 2, 3, 4, 5)

    mean, _ = fitter.weighted_mean(numbers, np.ones_like(numbers))

    assert mean == np.mean(numbers)


def test_weighted_mean():
    """
    Weighted mean with the same error on each point (i.e. no weights)

    """
    numbers = (1, 2, 3, 4, 5)
    errors = np.array((1.0, 0.5, 1.0, 1.0 / 3.0, 1.0))

    mean, _ = fitter.weighted_mean(numbers, errors)

    assert mean == 3.3125
