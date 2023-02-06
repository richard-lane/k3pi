"""
Efficiency unit test

"""
import sys
import pytest
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_efficiency"))

from lib_efficiency.metrics import _counts
from lib_efficiency.efficiency_util import scale_weights


def test_counts_underflow():
    """
    Check underflow gets caught

    """
    bins = [0, 1, 2, 3]
    x = [-0.5, 1.5, 2.5]
    wt = [1, 1, 1]

    with pytest.raises(ValueError):
        _counts(x, wt, bins)


def test_counts_overflow():
    """
    Check overflow gets caught

    """
    bins = [0, 1, 2, 3]
    x = [0.5, 1.5, 2.5, 3.5]
    wt = [1, 1, 1, 1]

    with pytest.raises(ValueError):
        _counts(x, wt, bins)


def test_counts():
    """
    Check that we get the counts and errors right when binning

    """
    bins = [0, 1, 2, 3]
    x = np.array([0.5, 0.5, 1.5, 2.5, 2.5])
    wt = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 2, 1.5]
    expected_errs = [np.sqrt(5), 2, np.sqrt(1.25)]

    counts, errs = _counts(x, wt, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_first_bin_empty():
    """
    See what happens when the first bin is empty

    """
    bins = [0, 1, 2, 3]
    x = np.array([1.5, 1.5, 1.5, 2.5, 2.5])
    wt = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [0, 5, 1.5]
    expected_errs = [0, 3, np.sqrt(1.25)]

    counts, errs = _counts(x, wt, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_last_bin_empty():
    """
    See what happens when the last bin is empty

    """
    bins = [0, 1, 2, 3]
    x = np.array([0.5, 0.5, 1.5, 1.5, 1.5])
    wt = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 3.5, 0]
    expected_errs = [np.sqrt(5), np.sqrt(5.25), 0.0]

    counts, errs = _counts(x, wt, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_middle_bin_empty():
    """
    See what happens when a bin in the middle is empty

    """
    bins = [0, 1, 2, 3]
    x = np.array([0.5, 0.5, 2.5, 2.5, 2.5])
    wt = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 0, 3.5]
    expected_errs = [np.sqrt(5), 0.0, np.sqrt(5.25)]

    counts, errs = _counts(x, wt, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_scale_wts():
    """
    Check we scale lists of weights correctly

    """
    # Make some lists of weights
    cf_wts = [np.array((2, 4, 8)), np.array((3, 4, 5)), np.array((6, 6, 6))]  # sum = 44
    dcs_wts = [
        np.array((1, 2, 3)),
        np.array((4, 5, 6)),
        np.array((2, 4, 6)),
    ]  # sum = 33

    # Ratio of efficiencies - this means for every 9 DCS evts
    # we reconstruct, we reconstruct 13.5 CF evts
    # This means we expect to scale the weights by 44/(33*1.5) = 8/9
    dcs_cf_ratio = 1.5

    # These are unchanged
    expected_cf = [
        np.array((2.0, 4.0, 8.0)),
        np.array((3.0, 4.0, 5.0)),
        np.array((6.0, 6.0, 6.0)),
    ]

    expected_dcs = [8 * arr / 9 for arr in dcs_wts]

    # Scale the weights
    cf_wts, dcs_wts = scale_weights(cf_wts, dcs_wts, dcs_cf_ratio)

    assert np.allclose(cf_wts[0], expected_cf[0])
    assert np.allclose(cf_wts[1], expected_cf[1])
    assert np.allclose(cf_wts[2], expected_cf[2])

    assert np.allclose(dcs_wts[0], expected_dcs[0])
    assert np.allclose(dcs_wts[1], expected_dcs[1])
    assert np.allclose(dcs_wts[2], expected_dcs[2])

    # Since we want the ratio of sums of weights to be
    # equal to the ratio of N generated evts,
    # this should be equal to the reciprocal of the ratio
    # of the efficiencies
    # (because N_gen = N_reco / eff)
    assert np.isclose(np.sum(dcs_wts) / np.sum(cf_wts), 1 / dcs_cf_ratio)
