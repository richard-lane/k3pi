"""
Things where I think it's actually useful to have a unit test

"""
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_fitter"))

from lib_time_fit import models, fitter, util


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


def test_abc():
    """
    Test we get the right abc params from constraint params

    """
    constraint_params = util.ConstraintParams(r_d=0.1, b=0.2, x=0.5, y=10.0)

    expected_abc = util.MixingParams(a=0.1, b=0.2, c=25.0625)
    actual_abc = models.abc(constraint_params)

    assert actual_abc.a == expected_abc.a
    assert actual_abc.b == expected_abc.b
    assert actual_abc.c == expected_abc.c


def test_constraint_params():
    """
    Test we get the right constraint params from scan params

    """
    scan_params = util.ScanParams(r_d=0.1, x=0.2, y=0.3, re_z=2.0, im_z=10.0)

    expected_constraint = util.ConstraintParams(r_d=0.1, b=2.6, x=0.2, y=0.3)
    actual_constraint = models.scan2constraint(scan_params)

    assert actual_constraint.r_d == expected_constraint.r_d
    assert actual_constraint.b == expected_constraint.b
    assert actual_constraint.x == expected_constraint.x
    assert actual_constraint.y == expected_constraint.y


def test_abc_scan():
    """
    Test we get the right abc params from scan params

    """
    scan_params = util.ScanParams(r_d=0.1, x=0.2, y=0.3, re_z=2.0, im_z=10.0)
    expected_abc = util.MixingParams(a=0.1, b=2.6, c=0.0325)

    actual_abc = models.abc_scan(scan_params)

    assert actual_abc.a == expected_abc.a
    assert actual_abc.b == expected_abc.b
    assert actual_abc.c == expected_abc.c
