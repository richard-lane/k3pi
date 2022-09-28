"""
Generate toy data, perform scans, find delta chi2 between the minimum and the true value

"""
import os
import sys
import time
import pathlib
from typing import Tuple
from multiprocessing import Manager, Process
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import common

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

from lib_time_fit import util, models, fitter


def _gen(
    domain: Tuple[float, float], params: util.ScanParams, widths, correlation
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some RS and WS times

    """
    n_rs = 5000000
    # TODO probably make the generator in outer scope
    gen = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    rs_t = common.gen_rs(gen, n_rs, domain)

    # Generate WS times drawing x and y from the Gaussian
    xy_vals = common.xy_vals(gen, 1, (params.x, params.y), widths, correlation)[0]
    ws_t = common.gen_ws(
        gen,
        n_rs,
        domain,
        models.abc_scan(
            util.ScanParams(params.r_d, *xy_vals, params.re_z, params.im_z)
        ),
    )

    return rs_t, ws_t


def _ratio_err(widths, correlation) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make some times, bin them, return their ratio and error

    Also return values for X and Y

    """
    # Define our fit parameters
    z = (0.75, -0.25)
    x_y = 0.06, 0.03
    params = util.ScanParams(1.0, *x_y, *z)

    # Generate some RS and WS times
    domain = 0.0, 8.0
    rs_t, ws_t = _gen(domain, params, widths, correlation)

    # Take their ratio in bins
    bins = np.linspace(*domain, 10)
    rs_count, rs_err = util.bin_times(rs_t, bins=bins)
    ws_count, ws_err = util.bin_times(ws_t, bins=bins)

    return (*util.ratio_err(ws_count, rs_count, ws_err, rs_err), params, bins)


def _coverage(out_dict: dict):
    """
    Put an array of delta chi2s in the dictionary

    """
    # Need x/y widths and correlations for the Gaussian constraint
    widths = (0.005, 0.005)
    correlation = 0.5
    n_experiments = 10

    delta_chi2 = np.ones(n_experiments) * np.inf
    for n in tqdm(range(n_experiments)):
        # Generate some times
        ratio, err, params, bins = _ratio_err(widths, correlation)

        # Perform a scan over them, find lowest chi2
        n_re, n_im = 50, 51
        allowed_rez = np.linspace(-1, 1, n_re)
        allowed_imz = np.linspace(-1, 1, n_im)
        chi2s = np.ones((n_im, n_re)) * np.inf

        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio, err, bins, these_params, widths, correlation
                )
                chi2s[j, i] = scan.fval

        # Perform a fit using the true value of Z
        true_chi2 = fitter.scan_fit(ratio, err, bins, params, widths, correlation).fval

        # Record delta chi2
        delta_chi2[n] = np.sqrt(abs(true_chi2 - np.min(chi2s)))

    out_dict[os.getpid()] = delta_chi2


def main():
    """
    Generate toy data, perform fits, show plots

    """
    out_dict = Manager().dict()

    n_procs = 8
    procs = [Process(target=_coverage, args=(out_dict,)) for _ in range(n_procs)]

    for p in procs:
        p.start()
        time.sleep(1)
    for p in procs:
        p.join()

    delta_chi2 = np.concatenate(list(out_dict.values()))

    fig, ax = plt.subplots()
    fig.suptitle("Coverage")
    ax.set_xlabel(r"$\sqrt{\Delta \chi^2}$")
    ax.set_ylabel(r"Frequency")
    bins = np.linspace(0, 3)
    ax.hist(
        delta_chi2,
        bins=bins,
        histtype="step",
        cumulative=True,
        weights=np.ones_like(delta_chi2) / len(delta_chi2),
    )
    ax.plot(bins, 2 * (norm.cdf(bins) - 0.5), "k--")
    fig.savefig("coverage.png")
    plt.show()


if __name__ == "__main__":
    main()
