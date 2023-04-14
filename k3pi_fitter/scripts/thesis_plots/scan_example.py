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
    n_rs = 2000000
    gen = np.random.default_rng()

    rs_t = common.gen_rs(gen, n_rs, domain)
    ws_t = common.gen_ws(gen, n_rs, domain, models.abc_scan(params))

    return rs_t, ws_t


def _ratio_err() -> Tuple[np.ndarray, np.ndarray]:
    """
    Make some times, bin them, return their ratio and error

    """
    # Define our fit parameters
    z = (0.75, -0.25)
    params = fit_util.ScanParams(
        0.055, 10 * definitions.CHARM_X, 10 * definitions.CHARM_Y, *z
    )

    # Generate some RS and WS times
    domain = 0.0, 10.0
    rs_t, ws_t = _gen(domain, params)

    # Take their ratio in bins
    bins = np.linspace(*domain, 20)
    rs_count, rs_err = stats.counts(values=rs_t, bins=bins)
    ws_count, ws_err = stats.counts(values=ws_t, bins=bins)

    return (*util.ratio_err(ws_count, rs_count, ws_err, rs_err), params, bins)


def _true_line_plot(ax: plt.Axes, params: fit_util.ScanParams):
    """
    Plot the expected relationship between best-fit ReZ and ImZ

    """
    points = np.linspace(*ax.get_xlim())
    expected = params.im_z + (params.y / params.x) * (params.re_z - points)
    ax.plot(points, expected, "y", linewidth=1, label="Expected")


def main():
    """
    Generate toy data, perform fits, show plots

    """
    # ratio we'll fit to
    ratio, err, params, bins = _ratio_err()

    n_re, n_im = 50, 51
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    fit_params = np.ones(chi2s.shape, dtype=object) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = fit_util.ScanParams(
                    params.r_d, params.x, params.y, re_z, im_z
                )
                scan = fitter.scan_fit(
                    ratio,
                    err,
                    bins,
                    these_params,
                    (definitions.CHARM_X_ERR, definitions.CHARM_Y_ERR),
                    definitions.CHARM_XY_CORRELATION,
                )
                fit_vals = scan.values
                fit_params[j, i] = fit_util.ScanParams(
                    r_d=fit_vals[0],
                    x=fit_vals[1],
                    y=fit_vals[2],
                    re_z=re_z,
                    im_z=im_z,
                )

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].set_xlim(bins[0], bins[-1])

    # Because plotting fcn doesn't deal with >3 colours at the moment TODO
    n_contours = 4
    contours = plotting.fits_and_scan(
        axes, (allowed_rez, allowed_imz), chi2s, fit_params, n_contours
    )

    widths = (bins[1:] - bins[:-1]) / 2
    centres = (bins[1:] + bins[:-1]) / 2
    axes[0].errorbar(centres, ratio, xerr=widths, yerr=err, fmt="k+")

    _true_line_plot(axes[1], params)
    axes[1].legend()

    # Plot true Z
    axes[1].plot(params.re_z, params.im_z, "y*")

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.savefig("toy_times_scan.png")


if __name__ == "__main__":
    main()
