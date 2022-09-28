"""
Pull study for fitter with no constraint

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_time_fit import util, models, fitter


def _plot(bins, ratio, err, actual_params, fit_params):
    """
    Plot

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    _, axis = plt.subplots()
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k+")

    fitted = [models.no_constraints(x, *fit_params) for x in centres]
    expected = [models.no_constraints(x, *actual_params) for x in centres]

    axis.plot(centres, expected, "k--", label="expected")
    axis.plot(centres, fitted, "r--", label="fitted")

    axis.set_ylabel("WS/RS")
    axis.set_xlabel(r"$t/\tau$")

    axis.legend()

    plt.show()


def _pull(
    rng: np.random.Generator, params: util.MixingParams, n_rs: int, plot: bool = False
):
    """
    Generate some events, perform a fit, work out the pulls

    """
    domain = (3, 10)
    bins = np.linspace(*domain, 30)

    rs_points = common.gen_rs(rng, n_rs, domain)
    ws_points = common.gen_ws(rng, n_rs, domain, params, plot=False)

    rs_count, rs_err = util.bin_times(rs_points, bins=bins)
    ws_count, ws_err = util.bin_times(ws_points, bins=bins)
    ratio, err = util.ratio_err(ws_count, rs_count, ws_err, rs_err)
    minuit = fitter.no_constraints(ratio, err, bins, params)

    if plot:
        _plot(bins, ratio, err, params, minuit.values)

    return (np.array(params) - minuit.values) / minuit.errors


def _plot_pull(a_pull, b_pull, c_pull):
    """
    Plot + show pulls

    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    bins = np.linspace(-3, 3, 50)
    ax[0].hist(a_pull, bins=bins)
    ax[1].hist(b_pull, bins=bins)
    ax[2].hist(c_pull, bins=bins)

    ax[0].set_title(f"{np.mean(a_pull):.4f}" + r"$\pm$" + f"{np.std(a_pull):.4f}")
    ax[1].set_title(f"{np.mean(b_pull):.4f}" + r"$\pm$" + f"{np.std(b_pull):.4f}")
    ax[2].set_title(f"{np.mean(c_pull):.4f}" + r"$\pm$" + f"{np.std(c_pull):.4f}")

    plt.show()


def main():
    """
    Generate RS and WS points, take their ratio, do some fits

    """
    rng = np.random.default_rng()

    n_rs = 6_000_00
    params = util.MixingParams(0.8, -0.05, 0.003)

    # Do a fit
    n_experiments = 100
    pulls = np.ones((3, n_experiments)) * np.nan
    for i in tqdm(range(n_experiments)):
        a_pull, b_pull, c_pull = _pull(rng, params, n_rs)

        pulls[0, i] = a_pull
        pulls[1, i] = b_pull
        pulls[2, i] = c_pull

    _plot_pull(pulls[0], pulls[1], pulls[2])


if __name__ == "__main__":
    main()
