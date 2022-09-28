"""
Things where I think it's actually useful to have a unit test

"""
import numpy as np
from .. import models


def test_rs_integral():
    """
    Check we get the right thing with a really simple test

    """
    assert np.allclose(models.rs_integral(np.array([0, np.inf])), np.array([1.0]))


def test_ws_integral():
    """
    Check we get the right integral by comparing with scipy
    numerical integral

    """
    lims = np.array([0.0, 1.0])
    args = 1.0, 2.0, 3.0

    expected_integral = np.array([9 - 20 / np.e])
    eval_integral = models.ws_integral(lims, *args)

    assert np.allclose(eval_integral, expected_integral)
