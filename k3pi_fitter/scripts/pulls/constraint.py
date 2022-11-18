"""
Pull study for fitter with Gaussian constraint on x and y

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_time_fit import util, models, fitter, plotting


def _plot(bins, ratio, err, actual_params, fit_params):
    """
    Plot

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    _, axis = plt.subplots()
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k+")

    plotting.constraints(
        axis, util.ConstraintParams(*fit_params), fmt="r--", label="Fit"
    )
    plotting.constraints(
        axis, util.ConstraintParams(*actual_params), fmt="k", label="Actual"
    )

    axis.set_ylabel("WS/RS")
    axis.set_xlabel(r"$t/\tau$")

    axis.legend()

    plt.show()


def _pull(
    rng: np.random.Generator,
    params: util.ConstraintParams,
    n_rs: int,
    xy_vals: Tuple[float, float],
    xy_mean_corr: Tuple[float, float, float],
    plot: bool = False,
):
    """
    Generate some events, perform a fit, work out the pulls

    """
    domain = (3, 10)
    bins = np.linspace(*domain, 30)

    these_params = util.ConstraintParams(params.r_d, params.b, *xy_vals)

    rs_points = common.gen_rs(rng, n_rs, domain)
    ws_points = common.gen_ws(rng, n_rs, domain, models.abc(these_params), plot=False)

    rs_count, rs_err = util.bin_times(rs_points, bins=bins)
    ws_count, ws_err = util.bin_times(ws_points, bins=bins)
    ratio, err = util.ratio_err(ws_count, rs_count, ws_err, rs_err)
    minuit = fitter.constraints(
        ratio, err, bins, params, xy_mean_corr[0:2], xy_mean_corr[2]
    )

    if plot:
        # Plot with the actual xy
        _plot(bins, ratio, err, these_params, minuit.values)

    # Calculate pull with the mean xy
    return (np.array(these_params) - minuit.values) / minuit.errors


def _plot_pull(pulls):
    """
    Plot + show pulls

    """
    fig, ax = plt.subplots(2, 2, figsize=(9, 3))
    bins = np.linspace(-3, 3, 50)
    ax[0, 0].hist(pulls[0], bins=bins)
    ax[0, 1].hist(pulls[1], bins=bins)
    ax[1, 0].hist(pulls[2], bins=bins)
    ax[1, 1].hist(pulls[3], bins=bins)

    params = r"$r_D$", "$b$", "$x$", "$y$"

    for pull, axis, label in zip(pulls, ax.ravel(), params):
        axis.set_title(f"{np.mean(pull):.4f}" + r"$\pm$" + f"{np.std(pull):.4f}")
        axis.set_xlabel(label)

    fig.tight_layout()
    plt.show()


def main():
    """
    Generate RS and WS points, take their ratio, do some fits

    """
    rng = np.random.default_rng()

    n_rs = 6_000_000
    params = util.ConstraintParams(1.0, 0.06, 0.03, 0.01)

    # Generate some random values of X and Y to use
    widths = np.array([0.01, 0.01])
    correlation = 0.1
    n_experiments = 20
    xy_vals = common.xy_vals(
        rng, n_experiments, (params.x, params.y), widths, correlation
    )

    # Do a fit
    pulls = np.ones((4, n_experiments)) * np.nan
    for i in tqdm(range(n_experiments)):
        r_pull, x_pull, y_pull, b_pull = _pull(
            rng, params, n_rs, xy_vals[i], (*widths, correlation), plot=False
        )

        pulls[0, i] = r_pull
        pulls[1, i] = x_pull
        pulls[2, i] = y_pull
        pulls[3, i] = b_pull

    _plot_pull(pulls)


if __name__ == "__main__":
    main()
