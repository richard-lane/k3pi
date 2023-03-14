"""
Generate some toy data, perform unconstrained fit

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))

from pulls import common
from lib_time_fit import util as fit_util, models, fitter, plotting
from lib_data import stats, util


def _gen(
    domain: Tuple[float, float], abc: fit_util.MixingParams
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
    axis: plt.Axes, bins: np.ndarray, ratio: np.ndarray, err: np.ndarray
) -> None:
    """
    Plot ratio and its error on an axis

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k.")


def main():
    """
    Generate toy data, perform fits, show plots

    """
    # Define our fit parameters
    params = fit_util.ScanParams(0.055, 0.06, 0.03, 0.25, -0.5)
    constraint_params = models.scan2constraint(params)
    abc_params = models.abc(constraint_params)

    # Generate some RS and WS times
    domain = 0.0, 10.0
    rs_t, ws_t = _gen(domain, abc_params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 30)
    rs_count, rs_err = stats.counts(rs_t, bins=bins)
    ws_count, ws_err = stats.counts(ws_t, bins=bins)

    ratio, err = util.ratio_err(ws_count, rs_count, ws_err, rs_err)

    # Perform fits to them
    no_constraint = fitter.no_constraints(ratio, err, bins, abc_params)

    # Plot fits
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    hist_kw = {"histtype": "step", "bins": bins}
    axes[0].hist(rs_t, **hist_kw, color="b", label="RS")
    axes[0].hist(ws_t, **hist_kw, color="g", label="WS")

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Counts")

    _plot_ratio(axes[1], bins, ratio, err)
    plotting.no_constraints(axes[1], abc_params, fmt="k-", label="True")
    plotting.no_constraints(
        axes[1], no_constraint.values, fmt="r--", label="No Constraint Fit"
    )

    axes[1].set_xlabel(r"$t/\tau$")
    axes[1].set_ylabel(r"$\frac{WS}{RS}$ ratio")

    for axis in axes:
        axis.legend()
        axis.set_xlim(bins[0], bins[-1])

    fig.savefig("toy_times.png")


if __name__ == "__main__":
    main()
