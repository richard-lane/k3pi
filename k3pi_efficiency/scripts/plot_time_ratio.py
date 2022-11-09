"""
Make plots of the ratio of decay times before/after reweighting

Test data

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common

from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)


def _times_and_weights(
    year: str,
    magnetisation: str,
    sign: str,
    data_sign: str,
    weighter_sign: str,
    fit: bool,
    cut: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get testing times and weights

    """
    if sign == "false":
        pgun_df = efficiency_util.pgun_df(sign, data_sign, train=False)
        ampgen_df = efficiency_util.ampgen_df("cf", data_sign, train=False)
    else:
        pgun_df = efficiency_util.pgun_df(sign, data_sign, train=False)
        ampgen_df = efficiency_util.ampgen_df(sign, data_sign, train=False)

    # Just pass the arrays into the efficiency function and it should find the right weights
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    ag_t, mc_t = ampgen_df["time"], pgun_df["time"]

    # Open the reweighter according to weighter_sign
    weights = efficiency_model.weights(
        mc_k,
        mc_pi1,
        mc_pi2,
        mc_pi3,
        mc_t,
        weighter_sign,
        year,
        "cf" if sign == "false" else sign,
        magnetisation,
        fit,
        cut,
        verbose=True,
    )

    return ag_t, mc_t, weights


def _allowed_times(
    times: np.ndarray, min_t: float, max_t: float, wt: np.ndarray = None
):
    mask = (min_t < times) & (times < max_t)
    if wt is None:
        return times[mask]
    return times[mask], wt[mask]


def main(args: argparse.Namespace):
    """
    Create a plot of decay times

    """
    bins = np.array(
        [
            efficiency_definitions.MIN_TIME,
            1.0,
            1.2,
            1.35,
            1.5,
            1.7,
            2.0,
            2.4,
            2.8,
            6.0,
            10.0,
        ]
    )
    rs_ag_t, rs_mc_t, rs_wt = _times_and_weights(
        args.year,
        args.magnetisation,
        "false",
        args.data_k_charge,
        args.weighter_k_charge,
        args.fit,
        args.cut,
    )
    ws_ag_t, ws_mc_t, ws_wt = _times_and_weights(
        args.year,
        args.magnetisation,
        "dcs",
        args.data_k_charge,
        args.weighter_k_charge,
        args.fit,
        args.cut,
    )

    # Keep only times in the allowed range
    rs_ag_t = _allowed_times(rs_ag_t, bins[0], bins[-1])
    ws_ag_t = _allowed_times(ws_ag_t, bins[0], bins[-1])

    rs_mc_t, rs_wt = _allowed_times(rs_mc_t, bins[0], bins[-1], rs_wt)
    ws_mc_t, ws_wt = _allowed_times(ws_mc_t, bins[0], bins[-1], ws_wt)

    plotting.plot_ratios(rs_mc_t, ws_mc_t, rs_ag_t, ws_ag_t, rs_wt, ws_wt, bins)

    fit_suffix = "_fit" if args.fit else ""
    plt.savefig(
        f"ratio_{args.year}_{args.magnetisation}_data_{args.data_k_charge}"
        f"_weighter_{args.weighter_k_charge}{fit_suffix}.png"
    )

    plt.show()


if __name__ == "__main__":
    parser = common.parser("Plot ratio of DCS/CF histograms")

    # Remove CF/DCS arguments, since we run over both DCS/CF
    # data and use both reweighters in this script
    common.remove_arg(parser, "decay_type")
    common.remove_arg(parser, "weighter_type")

    main(parser.parse_args())
