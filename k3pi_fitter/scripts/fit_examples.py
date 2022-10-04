"""
Generate some toy data, perform fits to it

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from pulls import common

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_time_fit import util, models, fitter, plotting


def _gen(
    domain: Tuple[float, float], abc: util.MixingParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some RS and WS times

    """
    n_rs = 24000000
    gen = np.random.default_rng()

    rs_t = common.gen_rs(gen, n_rs, domain)
    ws_t = common.gen_ws(gen, n_rs, domain, abc)

    return rs_t, ws_t


def _plot_ratio(
    ax: plt.Axes, bins: np.ndarray, ratio: np.ndarray, err: np.ndarray
) -> None:
    """
    Plot ratio and its error on an axis

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    ax.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.")


def main():
    """
    Generate toy data, perform fits, show plots

    """
    # Define our fit parameters
    params = util.ScanParams(0.055, 0.06, 0.03, 0.25, -0.5)
    constraint_params = models.scan2constraint(params)
    abc_params = models.abc(constraint_params)

    # Generate some RS and WS times
    domain = 0.0, 8.0
    rs_t, ws_t = _gen(domain, abc_params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 30)
    rs_count, rs_err = util.bin_times(rs_t, bins=bins)
    ws_count, ws_err = util.bin_times(ws_t, bins=bins)

    ratio, err = util.ratio_err(ws_count, rs_count, ws_err, rs_err)

    # Perform fits to them
    no_constraint = fitter.no_constraints(ratio, err, bins, abc_params)

    # Need x/y widths and correlations for the Gaussian constraint fit
    width = 0.01
    correlation = 0
    constraints = fitter.constraints(
        ratio, err, bins, constraint_params, (width, width), correlation
    )

    scan = fitter.scan_fit(ratio, err, bins, params, (width, width), correlation)

    # Plot fits
    fig, ax = plt.subplots()
    _plot_ratio(ax, bins, ratio, err)
    plotting.no_constraints(ax, abc_params, fmt="k-", label="True")
    plotting.no_constraints(
        ax, no_constraint.values, fmt="r--", label="No Constraint Fit"
    )

    plotting.constraints(
        ax,
        util.ConstraintParams(*constraints.values),
        fmt="b--",
        label="Constrained Fit",
    )

    plotting.scan_fit(
        ax,
        util.ScanParams(*scan.values, params.re_z, params.im_z),
        fmt="g--",
        label="Scan Fit",
    )

    ax.legend()
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$\frac{WS}{RS}$")
    fig.suptitle("Toy fits")

    fig.savefig("fit_examples.png")


if __name__ == "__main__":
    main()
