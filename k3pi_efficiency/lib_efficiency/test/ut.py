"""
Things where I think it's actually useful to have a unit test

"""
import pytest
import numpy as np
from ..metrics import _counts


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
