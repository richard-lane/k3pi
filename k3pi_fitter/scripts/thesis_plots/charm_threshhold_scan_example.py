"""
Generate some toy data, perform fits to it fixing Z
to different values

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))

from pulls import common
from lib_time_fit import util as fit_util, models, fitter, plotting, definitions
from lib_data import stats, util


def _gen(
    domain: Tuple[float, float], params: fit_util.ScanParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some RS and WS times

    """
    n_rs = 120_000_00
    gen = np.random.default_rng()

    rs_t = common.gen_rs(gen, n_rs, domain)
    ws_t = common.gen_ws(gen, n_rs, domain, models.abc_scan(params))

    return rs_t, ws_t


def _ratio_err() -> Tuple[np.ndarray, np.ndarray]:
    """
    Make some times, bin them, return their ratio and error

    """
    # Define our fit parameters
    z = (-0.27, -0.12)
    params = fit_util.ScanParams(0.055, definitions.CHARM_X, definitions.CHARM_Y, *z)

    # Generate some RS and WS times
    domain = 0.0, 10.0
    rs_t, ws_t = _gen(domain, params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 20)
    rs_count, rs_err = stats.counts(values=rs_t, bins=bins)
    ws_count, ws_err = stats.counts(values=ws_t, bins=bins)

    return (*util.ratio_err(ws_count, rs_count, ws_err, rs_err), params, bins)


def main():
    """
    Generate toy data, perform fits, show plots

    """
    # ratio we'll fit to
    ratio, err, params, bins = _ratio_err()

    n_re, n_im = 30, 31
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    phsp_bin = 3

    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_vals = np.ones((n_im, n_re), dtype=object) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.combined_fit(
                    ratio,
                    err,
                    bins,
                    these_params,
                    (definitions.CHARM_X_ERR, definitions.CHARM_Y_ERR),
                    definitions.CHARM_XY_CORRELATION,
                    phsp_bin,
                )

                chi2s[j, i] = scan.fval
                fit_vals[j, i] = fit_util.ScanParams(
                    r_d=scan.values[0],
                    x=scan.values[1],
                    y=scan.values[2],
                    re_z=re_z,
                    im_z=im_z,
                )
                pbar.update(1)

    chi2s -= np.nanmin(chi2s)
    chi2s = np.sqrt(chi2s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    widths = (bins[1:] - bins[:-1]) / 2
    centres = (bins[1:] + bins[:-1]) / 2
    axes[0].errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k+")

    n_contours = 4
    contours = plotting.fits_and_scan(
        axes, (allowed_rez, allowed_imz), chi2s, fit_vals, n_contours
    )

    # Plot true Z
    axes[1].plot(params.re_z, params.im_z, "y*")

    axes[1].set_xlabel(r"Re(Z)")
    axes[1].set_ylabel(r"Im(Z)")
    axes[1].add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.savefig("toy_scan_combined.png")


if __name__ == "__main__":
    main()
