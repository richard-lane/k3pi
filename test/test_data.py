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
from lib_data.util import flip_momenta, inv_mass, relative_angle


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
    pts = np.array([-0.5, 1.5, 2.5])

    # Test with the err handling fcn
    with pytest.raises(ValueError):
        stats._check_bins(np.digitize(pts, bins), len(bins) - 1, pts)

    # Also do an integration test with the actual fcn
    with pytest.raises(ValueError):
        stats.counts(pts, bins)


def test_counts_overflow():
    """
    Check overflow gets caught

    """
    bins = [0, 1, 2, 3]
    pts = np.array([0.5, 1.5, 2.5, 3.5])

    # Test with the err handling fcn
    with pytest.raises(ValueError):
        stats._check_bins(np.digitize(pts, bins), len(bins) - 1, pts)

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


def test_invmass():
    """
    Check we get the right invariant mass the sum of two particles

    """
    p_1 = np.array([[1], [2], [3], [10]])
    p_2 = np.array([[0.1], [0.2], [0.3], [5]])

    mass = np.sqrt(10403 / 50)

    assert inv_mass(p_1, p_2) == mass


def test_invmass_one_particle():
    """
    Check we get the right invariant mass of one particle

    """
    p_1 = np.array([[1], [2], [3], [10]])

    mass = np.sqrt(86)

    assert inv_mass(p_1) == mass


def test_invmass_arrays():
    """
    Check we get the right invariant masses when adding two arrays of particles

    """
    p_1 = np.array([[1, 2], [2, 3], [3, 4], [10, 11]])
    p_2 = np.array([[2, 4], [4, 6], [6, 8], [50, 51]])

    masses = np.array([np.sqrt(3474), np.sqrt(3583)])

    assert np.allclose(masses, inv_mass(p_1, p_2))


def test_integral_const_width():
    """
    Check integral given constant bin widths

    """
    points = np.linspace(0, 1, 1000)

    fcn = points**2

    assert np.isclose(
        stats.integral(points, fcn),
        1 / 3 * points[-1] ** 3,
    )


def test_integral_varying_width():
    """
    Check integral given different bin widths

    """
    rng = np.random.default_rng(seed=0)
    points = np.concatenate(([0], np.sort(rng.random(1000)), [1]))

    fcn = points**2

    assert np.isclose(
        stats.integral(points, fcn),
        1 / 3 * points[-1] ** 3,
    )


def test_angle_axes():
    """
    Check the angle between axes is right

    """
    x_axis = np.array([[1], [0], [0]])
    y_axis = np.array([[0], [1], [0]])

    expected_angle = 90

    angle = relative_angle(x_axis, y_axis)
    assert len(angle) == 1

    assert np.isclose(angle[0], expected_angle)


def test_angles():
    """
    Check the angle between vectors is right

    """
    x_axis = np.array([[1, 2], [0, 3], [4, 6]])
    y_axis = np.array([[4, 6], [2, 6], [5, 9]])

    expected_angles = np.array([29.805, 14.036])

    assert np.allclose(relative_angle(x_axis, y_axis), expected_angles, atol=0.001)
