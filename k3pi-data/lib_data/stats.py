"""
Statistical or related things

"""
import itertools
from typing import List, Tuple, Iterable
import numpy as np


def _indices(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """np.digitize - 1, basically"""
    return np.digitize(values, bins) - 1


def _sum_wts(
    n_bins: int, indices: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sum the weights in each bin

    Returns the sum of weights in each bin, and the sum of
    weights**2 in each bin (which should be sqrted to find
    the error)

    """
    count = np.ones(n_bins) * np.nan
    err = np.ones(n_bins) * np.nan

    for i in range(n_bins):
        mask = indices == i

        count[i] = np.sum(weights[mask])
        err[i] = np.sum(weights[mask] ** 2)

    return count, err


def _check_bins(indices: np.ndarray, n_bins: int, values: np.ndarray) -> None:
    """
    Check for under/overflow in bins given an array
    of indices as returned by np.digitize and the number
    of bins (len(bins) - 1, if the bins are all the left
    edges + the last bin's right edge)

    :param indices: array of bin indices
    :param n_bins: number of bins
    :param values: binned values; used to make the err output more useful

    :raises ValueError: in the case of under/overflow

    """
    # Underflow
    if -1 in indices:
        raise ValueError(f"Underflow: {values[indices == -1]}")

    # Overflow
    if n_bins in indices:
        raise ValueError(f"Overflow: {values[indices == n_bins]}")


def counts(
    values: np.ndarray, bins: np.ndarray, weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the counts in each bin and their errors

    :param values: values to count
    :param bins: Binning scheme;
                 left edge of each bin, plus the right edge of the highest bin
    :param weights: optional weights for each value

    :returns: array of counts in each bin
    :returns: array of errors in each bin

    """
    if weights is None:
        weights = np.ones_like(values)
    elif len(weights) != len(values):
        raise ValueError(f"{len(weights)=}\t{len(values)=}")

    indices = _indices(values, bins)
    n_bins = len(bins) - 1

    _check_bins(indices, n_bins, values)

    sum_wt, sum_wt_sq = _sum_wts(n_bins, indices, weights)

    return sum_wt, np.sqrt(sum_wt_sq)


def counts_generator(
    values: Iterable[np.ndarray], bins: np.ndarray, weights: Iterable[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the counts in each bin and their errors,
    given a generator of arrays of values and an optional
    generator of weights

    :param values: generator of values to count
    :param bins: Binning scheme;
                 left edge of each bin, plus the right edge of the highest bin
    :param weights: optional generator of weights for each value

    :returns: array of counts in each bin
    :returns: array of errors in each bin

    """
    # So we can zip arrays + weights even if weights
    # is not provided
    if weights is None:
        weights = itertools.repeat(None)

    # Init arrays
    n_bins = len(bins) - 1
    count = np.zeros(n_bins)
    err_sq = np.zeros(n_bins)

    for array, weight in zip(values, weights):
        indices = _indices(array, bins)
        _check_bins(indices, n_bins, array)

        # Add sum of weights to a histogram
        if weight is None:
            weight = np.ones_like(array)

        sum_wt, sum_wt_sq = _sum_wts(n_bins, indices, weight)

        count += sum_wt
        err_sq += sum_wt_sq

    # Take the square root of the errors
    return count, np.sqrt(err_sq)


def bin_indices(values: Iterable[np.ndarray], bins: np.ndarray) -> Iterable[np.ndarray]:
    """
    Generator of bin indices from a generator of values

    """
    n_bins = len(bins) - 1

    for array in values:
        indices = _indices(array, bins)
        _check_bins(indices, n_bins, array)

        yield indices


def time_binned_counts(
    values: Iterable[np.ndarray],
    bins: np.ndarray,
    n_time_bins: int,
    time_indices: Iterable[np.ndarray],
    weights: Iterable[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns the counts in each bin and their errors,
    given a time bin index for each value
    given a generator of arrays of values and an optional
    generator of weights

    :param values: generator of values to count
    :param bins: Binning scheme;
                 left edge of each bin, plus the right edge of the highest bin
    :param n_time_bins: number of time bins
    :param time_indices: array of time bin indices, one for each value
    :param weights: optional generator of weights for each value

    :returns: list of arrays of counts in each bin
    :returns: list of arrays of errors in each bin

    """
    # So we can zip arrays + weights even if weights
    # is not provided
    if weights is None:
        weights = itertools.repeat(None)

    # Init arrays
    n_bins = len(bins) - 1
    count = [np.zeros(n_bins) for _ in range(n_time_bins)]
    sum_wt_sq = [np.zeros(n_bins) for _ in range(n_time_bins)]

    for array, weight, time_indices_array in zip(values, weights, time_indices):
        # Add sum of weights to a histogram
        if weight is None:
            weight = np.ones_like(array)

        for i in range(n_time_bins):
            this_time_bin = time_indices_array == i

            indices = _indices(array[this_time_bin], bins)
            _check_bins(indices, n_bins, array[this_time_bin])

            sum_wt, sum_wt_sq_ = _sum_wts(n_bins, indices, weight[this_time_bin])

            count[i] += sum_wt
            sum_wt_sq[i] += sum_wt_sq_

    # Take the square root of the errors
    return count, [np.sqrt(array) for array in sum_wt_sq]
