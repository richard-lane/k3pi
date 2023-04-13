"""
Plot the ratio of pgun times before and after the D0
momentum weighting

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))

from lib_data import get, d0_mc_corrections, stats, util
from lib_time_fit.definitions import TIME_BINS


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

    t_bins = TIME_BINS[1:-1]

    # Only keep points in the time bins
    ws_keep = (t_bins[0] < ws_t) & (ws_t < t_bins[-1])
    rs_keep = (t_bins[0] < rs_t) & (rs_t < t_bins[-1])

    ws_weights = ws_weights[ws_keep]
    ws_t = ws_t[ws_keep]

    rs_weights = rs_weights[rs_keep]
    rs_t = rs_t[rs_keep]

    # Get ratio before
    ws_before = stats.counts(ws_t, t_bins)
    rs_before = stats.counts(rs_t, t_bins)

    ws_after = stats.counts(ws_t, t_bins, ws_weights)
    rs_after = stats.counts(rs_t, t_bins, rs_weights)

    fig, axis = plt.subplots()
    centres = (t_bins[1:] + t_bins[:-1]) / 2
    widths = t_bins[1:] - t_bins[:-1]

    before = util.ratio_err(ws_before[0], rs_before[0], ws_before[1], rs_before[1])
    after = util.ratio_err(ws_after[0], rs_after[0], ws_after[1], rs_after[1])

    axis.errorbar(
        centres, before[0], xerr=widths / 2, yerr=before[1], label="No Weight", fmt="."
    )
    axis.errorbar(
        centres,
        after[0],
        xerr=widths / 2,
        yerr=after[1],
        label="D0 Momentum Weighted",
        fmt=".",
    )

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
