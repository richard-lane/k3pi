"""
k3pi-data unit test

"""
import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))

from lib_data import definitions
from lib_data import stats
from lib_data.util import flip_momenta


def test_flip_momenta():
    """
    Check we flip momenta correctly

    """
    # Make a toy dataframe
    n_labels = len(definitions.MOMENTUM_COLUMNS)
    dataframe = pd.DataFrame(
        dict(zip(definitions.MOMENTUM_COLUMNS, [[l, l * 2] for l in range(n_labels)]))
    )

    # expected df after flipping k3pi 3 momenta from the first row
    to_flip = np.array([True, False], dtype=np.bool_)
    expected_df = dataframe.copy()

    # change sign for k3pi 3 momenta in the first row
    k3pi_3momenta = expected_df.columns.str.contains(
        "us"
    ) & ~expected_df.columns.str.endswith("_E")
    expected_df.loc[0, k3pi_3momenta] *= -1

    # check that they're the same
    dataframe = flip_momenta(dataframe, to_flip=to_flip)
    assert dataframe.iloc[0].equals(expected_df.iloc[0])
    assert dataframe.iloc[1].equals(expected_df.iloc[1])


def test_wrong_number_weights():
    """
    Check an error gets caught if the weights/values
    have different lengths

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 0.5, 1.5, 2.5, 2.5])
    weights = np.array([1.0, 2.0, 2.0, 0.5])

    with pytest.raises(ValueError):
        stats.counts(pts, bins, weights)


def test_counts_underflow():
    """
    Check underflow gets caught

    """
    bins = [0, 1, 2, 3]
    pts = [-0.5, 1.5, 2.5]

    # Test with the err handling fcn
    with pytest.raises(ValueError):
        stats._check_bins(np.digitize(pts, bins), len(bins) - 1)

    # Also do an integration test with the actual fcn
    with pytest.raises(ValueError):
        stats.counts(pts, bins)


def test_counts_overflow():
    """
    Check overflow gets caught

    """
    bins = [0, 1, 2, 3]
    pts = [0.5, 1.5, 2.5, 3.5]

    # Test with the err handling fcn
    with pytest.raises(ValueError):
        stats._check_bins(np.digitize(pts, bins), len(bins) - 1)

    # Also do an integration test with the actual fcn
    with pytest.raises(ValueError):
        stats.counts(pts, bins)


def test_counts():
    """
    Check we get the right counts and errors when binning

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 0.5, 1.5, 2.5, 2.5])

    expected_counts = [2, 1, 2]
    expected_errs = [np.sqrt(2), 1, np.sqrt(2)]

    counts, errs = stats.counts(pts, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_counts_weighted():
    """
    Check that we get the counts and errors right when binning

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 0.5, 1.5, 2.5, 2.5])
    weights = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 2, 1.5]
    expected_errs = [np.sqrt(5), 2, np.sqrt(1.25)]

    counts, errs = stats.counts(pts, bins, weights)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_first_bin_empty():
    """
    See what happens when the first bin is empty

    """
    bins = [0, 1, 2, 3]
    pts = np.array([1.5, 1.5, 1.5, 2.5, 2.5])
    weights = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [0, 5, 1.5]
    expected_errs = [0, 3, np.sqrt(1.25)]

    counts, errs = stats.counts(pts, bins, weights)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_last_bin_empty():
    """
    See what happens when the last bin is empty

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 0.5, 1.5, 1.5, 1.5])
    weights = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 3.5, 0]
    expected_errs = [np.sqrt(5), np.sqrt(5.25), 0.0]

    counts, errs = stats.counts(pts, bins, weights)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_middle_bin_empty():
    """
    See what happens when a bin in the middle is empty

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 0.5, 2.5, 2.5, 2.5])
    weights = np.array([1.0, 2.0, 2.0, 0.5, 1.0])

    expected_counts = [3, 0, 3.5]
    expected_errs = [np.sqrt(5), 0.0, np.sqrt(5.25)]

    counts, errs = stats.counts(pts, bins, weights)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_counts_generator():
    """
    Check we get the right counts from a generator

    """
    bins = [0, 1, 2, 3]
    pts = (np.array([0.5, 0.5, 1.5, 2.5, 2.5]) for _ in range(2))

    expected_counts = [4, 2, 4]
    expected_errs = [2.0, np.sqrt(2), 2.0]

    counts, errs = stats.counts_generator(pts, bins)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)


def test_counts_generator_weighted():
    """
    Check we get the right counts from a generator

    """
    bins = [0, 1, 2, 3]
    pts = (np.array([0.5, 0.5, 1.5, 2.5, 2.5]) for _ in range(2))
    weights = (np.array([1.0, 2.0, 2.0, 0.5, 1.0]) for _ in range(2))

    expected_counts = [6, 4, 3]
    expected_errs = [np.sqrt(10), np.sqrt(8), np.sqrt(2.5)]

    counts, errs = stats.counts_generator(pts, bins, weights)

    assert np.all(expected_counts == counts)
    assert np.all(expected_errs == errs)
