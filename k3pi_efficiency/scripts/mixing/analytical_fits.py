"""
Use some made-up PDFs to introduce a known amount of mixing with
known toy PDFs

Lots of things (the pdfs, their integrals) are hard coded, so you can't just
change the PDF and expect everything to work
(you'll have to change the integral and interference functions too).

"""
import sys
import pathlib
from typing import Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))

import pdg_params
from lib_efficiency import mixing
from lib_time_fit import util, fitter, plotting


# PDFs domain
DOMAIN = (0, np.pi)

# PDFS should always have mod^2 < this
PDF_MAX = 1


def _generate(
    rng: np.random.Generator, n_gen: int, pdf: Callable[[float], complex]
) -> np.ndarray:
    """
    Generate evts from a complex PDF

    n_gen generated; returns fewer than this because of the accept-reject sampling

    """
    pts = DOMAIN[0] + DOMAIN[1] * rng.random(n_gen)
    pdf_vals = pdf(pts)
    acceptance_prob = np.abs(pdf_vals) ** 2

    assert acceptance_prob.max() <= PDF_MAX, acceptance_prob.max()

    return pts[rng.random(n_gen) * PDF_MAX < acceptance_prob]


def _d0(x: float) -> complex:
    """
    RS PDF, defined between 0 and pi

    """
    return np.sin(x) + 0.5j * np.cos(x)


def _dbar0(x: float) -> complex:
    """
    WS PDF, defined between 0 and pi

    """
    return np.exp(-x) / 50.0 + 1.0j * np.exp(x) / 75.0


def _d0_prob(x: float) -> float:
    """
    Not normalised

    """
    return np.abs(_d0(x)) ** 2


def _dbar0_prob(x: float) -> float:
    """
    Not normalised

    """
    return np.abs(_dbar0(x)) ** 2


def _d0_integral() -> complex:
    """
    Analytical integral of |_d0|^2 over the whole domain

    """
    return 5.0 * np.pi / 8.0


def _dbar0_integral() -> complex:
    """
    Analytical integral of |_dbar0|^2 over the whole domain

    """
    return (5 - 9 * np.exp(-2 * np.pi) + 4 * np.exp(2 * np.pi)) / 45000


def _interference_z() -> complex:
    """
    Expected interference factor given the amplitudes

    """
    # real and imag parts
    real = (2 - np.sinh(np.pi) + 2 * np.cosh(np.pi)) / 150
    imag = (-1 - 7 * np.sinh(np.pi) - np.cosh(np.pi)) / 600

    # Scale
    return (real + 1.0j * imag) / np.sqrt(_dbar0_integral() * _d0_integral())


def _plot_pdfs():
    """
    Plot and save pdfs

    """
    fig, (d0_ax, dbar0_ax) = plt.subplots(1, 2, figsize=(10, 5))

    x_vals = np.linspace(*DOMAIN)

    d0_vals = _d0(x_vals)
    dbar0_vals = _dbar0(x_vals)

    real_kw = {"color": "blue", "label": "Real"}
    imag_kw = {"color": "plum", "label": "Imag"}
    prob_kw = {"color": "k", "label": "Prob", "linestyle": "--"}

    d0_ax.plot(x_vals, d0_vals.real, **real_kw)
    d0_ax.plot(x_vals, d0_vals.imag, **imag_kw)
    d0_ax.plot(x_vals, _d0_prob(x_vals), **prob_kw)

    dbar0_ax.plot(x_vals, dbar0_vals.real, **real_kw)
    dbar0_ax.plot(x_vals, dbar0_vals.imag, **imag_kw)
    dbar0_ax.plot(x_vals, _dbar0_prob(x_vals), **prob_kw)

    d0_ax.set_title(r"$D^0$")
    dbar0_ax.set_title(r"$\overline{D}^0$")

    d0_ax.legend()
    fig.tight_layout()

    fig.savefig("analytical_fit_pdfs.png")
    plt.close(fig)


def _plot_hists(d0_pts: np.ndarray, dbar0_pts: np.ndarray):
    """
    Plot histograms of events and the mod squared of pdfs

    """
    fig, (d0_ax, dbar0_ax) = plt.subplots(2, 1, figsize=(10, 5))
    x_vals = np.linspace(*DOMAIN, 100)
    hist_kw = {"histtype": "step", "bins": x_vals}
    bin_width = x_vals[1] - x_vals[0]

    d0_ax.hist(d0_pts, **hist_kw)
    dbar0_ax.hist(dbar0_pts, **hist_kw)

    d0_ax.plot(x_vals, bin_width * len(d0_pts) * _d0_prob(x_vals) / _d0_integral())
    dbar0_ax.plot(
        x_vals,
        bin_width * len(dbar0_pts) * _dbar0_prob(x_vals) / _dbar0_integral(),
    )

    d0_ax.set_title(r"$D^0$")
    dbar0_ax.set_title(r"$\overline{D}^0$")

    fig.tight_layout()
    fig.savefig("analytical_fit_hists.png")
    plt.close(fig)


def _times(rng: np.random.Generator, num: int) -> np.ndarray:
    """
    Decay times, no mixing (i.e. an exponential)

    """
    return rng.exponential(scale=1, size=num)


def _rd():
    """Amplitude ratio"""
    return np.sqrt(_dbar0_integral() / _d0_integral())


def _expected_ratio(times: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Time-dependent ratio given the mixing params x and y

    """
    r_d = _rd()
    z = _interference_z()

    const = r_d**2
    linear = r_d * ((x * z.imag) + (y * z.real))
    quadratic = (x**2 + y**2) / 4.0

    return const + linear * times + quadratic * times**2


def _exact_expected(times: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Time dependent ratio, without assuming x and y are small

    """
    r_d = _rd()
    z = _interference_z()

    yt = y * times
    xt = x * times

    return 0.5 * (
        r_d**2 * (np.cosh(yt) + np.cos(xt))
        + (np.cosh(yt) - np.cos(xt))
        + 2 * r_d * (z.real * np.sinh(yt) + z.imag * np.sin(xt))
    )


def _plot_mixed_dists(
    dbar0_times: np.ndarray, dbar0_pts: np.ndarray, weights: np.ndarray
):
    """
    Plot time and phase space distributions with and without mixing

    """
    fig, (time_ax, phsp_ax) = plt.subplots(1, 2, figsize=(10, 5))

    plot_kw = {"histtype": "step"}
    time_kw = {"bins": np.linspace(0, 15, 100), **plot_kw}
    time_ax.hist(dbar0_times, **time_kw, color="k", label="No mixing")
    time_ax.hist(dbar0_times, **time_kw, weights=weights, color="r", label="Mixing")

    phsp_kw = {"bins": np.linspace(0, np.pi, 100), **plot_kw}
    phsp_ax.hist(dbar0_pts, **phsp_kw, color="k")
    phsp_ax.hist(dbar0_pts, **phsp_kw, weights=weights, color="r")

    time_ax.set_xlabel(r"$t/\tau$")
    phsp_ax.set_xlabel(r"$x$")

    time_ax.legend()

    fig.tight_layout()
    fig.savefig("analytical_fit_mixed_dists.png")
    plt.close(fig)


def _plot_ratio(
    bins: np.ndarray,
    ratio: np.ndarray,
    err: np.ndarray,
    weighted_ratio: np.ndarray,
    weighted_err: np.ndarray,
    params: mixing.MixingParams,
):
    """
    Plot ratio of decay times

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    fig, axis = plt.subplots()

    axis.errorbar(centres, ratio, xerr=widths, yerr=err, label="No Mixing", fmt="k+")
    axis.errorbar(
        centres,
        weighted_ratio,
        xerr=widths,
        yerr=weighted_err,
        label="Mixing",
        fmt="r+",
    )

    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, _expected_ratio(pts, 0, 0), "k:", label="No Mixing")
    axis.plot(
        pts,
        _exact_expected(pts, params.mixing_x, params.mixing_y),
        "--",
        color="plum",
        label="Mixing Exact",
    )
    axis.plot(
        pts,
        _expected_ratio(pts, params.mixing_x, params.mixing_y),
        "r:",
        label="Mixing Quadratic",
    )

    axis.legend()
    axis.set_xlabel(r"$t/\tau$")
    axis.set_ylabel(r"$\frac{\overline{D}^0}{D^0}$")
    fig.tight_layout()
    fig.savefig("analytical_fit_ratio.png")
    plt.close(fig)


def _plot_scan(
    allowed_rez: np.ndarray,
    allowed_imz: np.ndarray,
    chi2s: np.ndarray,
    n_levels: int,
    true_z: Tuple[float, float],
):
    """
    plot a scan

    """
    fig, axis = plt.subplots()

    contours = plotting.scan(
        axis,
        allowed_rez,
        allowed_imz,
        chi2s,
        levels=np.arange(n_levels),
    )

    # Plot the true/generating value of Z
    axis.plot(*true_z, "y*")

    axis.set_xlabel(r"Re(Z)")
    axis.set_ylabel(r"Im(Z)")
    axis.add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    fig.savefig("analytical_fit_scan.png")
    plt.close(fig)


def _scan(
    ratio: np.ndarray,
    err: np.ndarray,
    bins: np.ndarray,
    x: float,
    y: float,
    r_d: float,
    z: complex,
):
    """
    Plot a scan for the fits

    """
    # Need x/y widths and correlations for the Gaussian constraint
    correlation = 0.0

    n_re, n_im = 51, 50
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf
    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                these_params = util.ScanParams(r_d, x, y, re_z, im_z)
                scan = fitter.scan_fit(
                    ratio, err, bins, these_params, (x / 10, y / 10), correlation
                )

                chi2s[j, i] = scan.fval
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    _plot_scan(allowed_rez, allowed_imz, chi2s, n_levels=7, true_z=(z.real, z.imag))


def _add_mixing(
    dbar0_times: np.ndarray,
    d0_times: np.ndarray,
    dbar0_pts: np.ndarray,
):
    """
    Add some mixing, plot ratio of times

    """
    # Add some mixing
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=0.003,
        mixing_y=0.006,
    )

    # Need to pass the amplitudes in in the right order
    # it should be obvious if you've done this wrong
    weights = mixing._ws_weights(
        dbar0_times,
        _dbar0(dbar0_pts),
        _d0(dbar0_pts),
        params,
        (1 / np.sqrt(2), 1 / np.sqrt(2)),
    )

    # Plot phase space and time distributions before and
    # after mixing
    _plot_mixed_dists(dbar0_times, dbar0_pts, weights)

    # Plot ratio of points before/after mixing
    bins = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 11, 13])

    d0_count, _ = np.histogram(d0_times, bins=bins)
    dbar0_count, _ = np.histogram(dbar0_times, bins=bins)
    weighted_count, _ = np.histogram(dbar0_times, weights=weights, bins=bins)

    ratio = dbar0_count / d0_count
    weighted_ratio = weighted_count / d0_count

    err = ratio * np.sqrt(1 / d0_count + 1 / dbar0_count)
    weighted_err = weighted_ratio * np.sqrt(1 / d0_count + 1 / weighted_count)

    _plot_ratio(bins, ratio, err, weighted_ratio, weighted_err, params)

    # Do a scan
    _scan(
        weighted_ratio,
        weighted_err,
        bins,
        params.mixing_x,
        params.mixing_y,
        ratio[0],  # The unweighted ratio will be approx= r_D
        _interference_z(),
    )


def main():
    """
    Generate some events with our toy PDFs,
    add some mixing to them,
    plot the ratio of WS to WS with the expected trend

    """
    # Plot PDFs
    _plot_pdfs()

    # Generate D0 and Dbar0 points
    generator = np.random.default_rng()
    n_gen = 2_000_000
    d0_pts = []
    dbar0_pts = []
    for _ in tqdm(range(20)):
        d0_pts.append(_generate(generator, n_gen, _d0))
        dbar0_pts.append(_generate(generator, n_gen, _dbar0))
    d0_pts = np.concatenate(d0_pts)
    dbar0_pts = np.concatenate(dbar0_pts)

    print(f"{len(d0_pts):,}")
    print(f"{len(dbar0_pts):,}")

    # Plot histograms
    _plot_hists(d0_pts, dbar0_pts)

    # Generate times (exponential)
    d0_times = _times(generator, len(d0_pts))
    dbar0_times = _times(generator, len(dbar0_pts))

    # Plot also the expected ratio
    _add_mixing(dbar0_times, d0_times, dbar0_pts)


if __name__ == "__main__":
    main()
