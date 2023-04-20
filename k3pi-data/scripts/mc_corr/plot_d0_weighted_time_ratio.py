"""
Plot the ratio of pgun times before and after the D0
momentum weighting

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_efficiency"))

from lib_data import get, d0_mc_corrections, stats, util
from lib_time_fit.definitions import TIME_BINS
from lib_efficiency.plotting import _plot_shaded_area


def _line(pts: np.ndarray, a: float, b: float) -> np.ndarray:
    """Straight line"""
    return a * pts + b


def main(*, year: str, magnetisation: str):
    """
    Plot stuff

    """
    # Get testing particle gun dataframes
    ws_df = get.particle_gun(year, "dcs", magnetisation, show_progress=True)
    ws_df = ws_df[~ws_df["train"]]

    rs_df = get.particle_gun(year, "cf", magnetisation, show_progress=True)
    rs_df = rs_df[~rs_df["train"]]

    # Get D0 MC corr weights
    rs_weights = d0_mc_corrections.get_pgun(year, "cf", magnetisation).weights(
        d0_mc_corrections.d0_points(rs_df)
    )
    rs_weights /= np.mean(rs_weights)
    ws_weights = d0_mc_corrections.get_pgun(year, "dcs", magnetisation).weights(
        d0_mc_corrections.d0_points(ws_df)
    )
    ws_weights /= np.mean(ws_weights)

    # Get times
    ws_t = ws_df["time"]
    rs_t = rs_df["time"]

    t_bins = TIME_BINS[2:-1]

    # Only keep points in the time bins
    ws_keep = (t_bins[0] < ws_t) & (ws_t < t_bins[-1])
    rs_keep = (t_bins[0] < rs_t) & (rs_t < t_bins[-1])

    ws_weights = ws_weights[ws_keep]
    ws_t = ws_t[ws_keep]

    rs_weights = rs_weights[rs_keep]
    rs_t = rs_t[rs_keep]

    # Scale WS to expected stats in each bin by throwing points away
    rng = np.random.default_rng()
    # time-bin integrated yield
    expected_ws_yield = 2000 * (len(t_bins) - 1)
    keep_frac = expected_ws_yield / len(ws_t)
    print(f"{keep_frac=}")
    keep = rng.random(len(ws_t)) < keep_frac
    ws_t = ws_t[keep]
    ws_weights = ws_weights[keep]

    # Scale RS weights to give expected RS stats
    expected_rs_yield = expected_ws_yield / 0.055**2
    rs_scale_wts = np.full(len(rs_t), expected_rs_yield / len(rs_t))

    # Get ratio before
    ws_before = stats.counts(ws_t, t_bins)
    rs_before = stats.counts(rs_t, t_bins, rs_scale_wts)

    before = util.ratio_err(ws_before[0], rs_before[0], ws_before[1], rs_before[1])

    # Get ratio after
    ws_after = stats.counts(ws_t, t_bins, ws_weights)
    rs_after = stats.counts(rs_t, t_bins, rs_weights * rs_scale_wts)

    after = util.ratio_err(ws_after[0], rs_after[0], ws_after[1], rs_after[1])

    # Fit straight lines to them
    centres = (t_bins[1:] + t_bins[:-1]) / 2
    widths = t_bins[1:] - t_bins[:-1]
    before_opt, before_cov = curve_fit(
        _line,
        centres,
        before[0],
        sigma=before[1],
        absolute_sigma=True,
        p0=(0, 0.055**2),
    )
    after_opt, after_cov = curve_fit(
        _line,
        centres,
        after[0],
        sigma=after[1],
        absolute_sigma=True,
        p0=(0, 0.055**2),
    )

    fig, axis = plt.subplots()

    axis.errorbar(
        centres,
        before[0],
        xerr=widths / 2,
        yerr=before[1],
        label="No Weight",
        fmt=".",
        color="k",
    )
    axis.errorbar(
        centres,
        after[0],
        xerr=widths / 2,
        yerr=after[1],
        label="D0 Momentum Weighted",
        fmt=".",
        color="r",
    )

    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, _line(pts, *before_opt), "k")
    axis.plot(pts, _line(pts, *after_opt), "r")

    _plot_shaded_area(axis, before_opt, before_cov, pts)
    _plot_shaded_area(axis, after_opt, after_cov, pts)

    axis.legend()
    fig.tight_layout()

    fig.suptitle("Testing Data")

    fig.tight_layout()

    path = "d0_distributions_time_ratio.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )

    main(**vars(parser.parse_args()))
