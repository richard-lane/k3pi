"""
Example script for how to use the reweighter

The phase space reweighting is handled by a hep_ml.reweight.GBReweighter
The time reweighting is handled by either a histogram division,
or by performing a fit to the decay times

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar library
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_efficiency import time_fitter


def _target_points(gen: np.random.Generator, n_gen: int) -> np.ndarray:
    """
    Make up some points - target distribution

    "phase space" part are just gaussians, time part looks like
    an exponential

    """
    # Phsp part is just 5 gaussians, maybe with correlations
    mean = [0.5, -0.5, 0, 0.5, -0.5]
    cov = np.diag([1, 1, 1.2, 1, 0.8])
    cov[1, 2] = 0.5
    cov[2, 1] = 0.5

    phsp = gen.multivariate_normal(mean, cov, n_gen)

    # Time part is an exponential
    time = gen.exponential(size=n_gen)

    return np.column_stack((phsp, time))


def pdf(time: float, params: Tuple):
    """ PDF of efficiency(t) * e^t """
    # Have to convert to an array/back for annoying reasons
    return time_fitter.normalised_pdf(np.array([time]), *params)[1][0]


def _orig_points(gen: np.random.Generator, n_gen: int) -> np.ndarray:
    """
    Make up some points

    "phase space" part are just gaussians, time part looks like
    an exponential * an efficiency

    """
    # Phsp part is just 5 gaussians, maybe with correlations
    mean = [0, -0.1, 0.1, 0.5, -0.5]
    cov = np.diag([1, 1.1, 1.2, 0.9, 0.8])
    cov[1, 2] = 0.6
    cov[2, 1] = 0.6

    phsp = gen.multivariate_normal(mean, cov, n_gen)

    # Parameters used for the time-efficiency function
    # It doesn't really matter what these do, but they give
    # a vaguely realistic looking efficiency
    params = (0.21, 1.2, 2.2, 1.3, 1.45, 0.99)

    # Perform accept-reject to get the right decay times
    # python loop - could speed up using numpy but that
    # might make the code confusing
    times = np.empty(n_gen)
    n_generated = 0
    max_lifetimes = 8
    max_pdf = 0.5
    with tqdm(total = n_gen) as pbar:
        while n_generated < n_gen:
            x = max_lifetimes * gen.random()
            y = max_pdf * gen.random()

            if y < pdf(x, params):
                times[n_generated] = x
                n_generated += 1
                pbar.update(1)

    return np.column_stack((phsp, times))


def _plot(
    target: np.ndarray, original: np.ndarray, weights: np.ndarray, min_t
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distributions before/after reweighting

    :param target: (N, 6) shape array of target points
    :param original: (N, 6) shape array of original points
    :param weights: weights to apply to original to make it look like target
    :param min_t: minimum time for plotting
    :returns: matplotlib stuff

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"histtype": "step", "density": True}
    # Plot phsp
    for axis, a, b in zip(ax.ravel(), original.T[:-1], target.T[:-1]):
        _, bins, _ = axis.hist(a, bins=50, label="Original", **hist_kw)
        axis.hist(b, bins=bins, label="Target", **hist_kw)
        if weights is not None:
            axis.hist(a, bins=bins, label="Reweighted", weights=weights, **hist_kw)

    # Plot time
    # Only plot above the min time to make the hists scaled the same
    a = original.T[-1]
    b = target.T[-1]

    keep_orig = a > min_t
    keep_target = b > min_t
    a = a[keep_orig]
    b = b[keep_target]

    time_ax = ax.ravel()[-1]
    _, bins, _ = time_ax.hist(a, bins=50, label="Original", **hist_kw)
    time_ax.hist(b, bins=bins, label="Target", **hist_kw)
    if weights is not None:
        w = weights[keep_orig]
        time_ax.hist(a, bins=bins, label="Reweighted", weights=w, **hist_kw)

    fig.tight_layout()
    ax.ravel()[-1].legend()

    return fig, ax


def main():
    """
    Create some made-up distributions
    Use the reweighter to reweight one to the other
    Plot

    """
    # Make up some data
    # In reality, train the reweighter on (N, 6) shape arrays of
    # (x1, x2, x3, x4, x5, t) where each x is a phase space variable
    # (e.g. using fourbody.param.helicity_param)
    gen = np.random.default_rng()
    n_gen = 50000
    print("generating toys")
    target = _target_points(gen, n_gen)
    original = _orig_points(gen, n_gen)

    # Keyword args to pass to GBReweighter
    train_kwargs = {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.2,
        "min_samples_leaf": 200,
    }

    # Create a reweighter with a fit
    # This reweighter will set all weights below the min time to 0, but perform the
    # fit using all times
    min_t = 0.45
    print("\ncreating time fit reweighter")
    fit_weighter = EfficiencyWeighter(
        target, original, fit=True, min_t=min_t, **train_kwargs
    )

    # Create a reweighter without a fit (do a histogram division instead)
    # This reweighter will set weights below the min time to 0
    print("\ncreating hist division reweighter")
    histogram_weighter = EfficiencyWeighter(
        target,
        original,
        fit=False,
        min_t=min_t,
        n_bins=1000,  # Sensible value will give you ~20 evts in each bin
        n_neighs=10.0,  # Sensible value depends on n_bins
        **train_kwargs
    )

    # Plot time fit weighting
    fit_weights = fit_weighter.weights(original)
    fig, ax = _plot(target, original, fit_weights, min_t)

    # Hacky - plot the fit
    pts = np.linspace(*ax.ravel()[-1].get_xlim())
    params = fit_weighter._time_weighter.fitter.fit_vals
    ax.ravel()[-1].plot(pts, pdf(pts, params), "r--")

    fig.suptitle("Reweight with time fit")
    fig.tight_layout()
    plt.show()

    # Plot hist division weighting
    hist_weights = histogram_weighter.weights(original)
    fig, _ = _plot(target, original, hist_weights, min_t)
    fig.suptitle("Reweight with time histogram division")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
