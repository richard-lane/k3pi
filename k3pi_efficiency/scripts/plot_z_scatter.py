"""
Make scatter plots of the numerically evaluated coherence factor

Test data

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

import common
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
)
from lib_efficiency.get import reweighter_dump

from lib_data import d0_mc_corrections
from lib_cuts.get import classifier as get_clf, signal_cut_df
from lib_cuts.definitions import THRESHOLD


def main(args: argparse.Namespace):
    """
    Create a plot

    """
    pgun_df = efficiency_util.pgun_df(
        args.year, args.magnetisation, args.decay_type, args.data_k_charge, train=False
    )
    ampgen_df = efficiency_util.ampgen_df(
        args.decay_type, args.data_k_charge, train=False
    )
    # We might want to do BDT cut
    if args.cut:
        # Always use DCS classifier for BDT cut, even on RS data
        pgun_df = signal_cut_df(
            pgun_df, get_clf(args.year, "dcs", args.magnetisation), THRESHOLD
        )

    reweighter = reweighter_dump(
        args.year,
        args.weighter_type,
        args.magnetisation,
        args.weighter_k_charge,
        args.fit,
        args.cut,
        verbose=True,
    )
    weights = efficiency_util.wts_df(pgun_df, reweighter)

    # Get D0 MC corr wts
    mc_corr_wt = d0_mc_corrections.pgun_wt_df(pgun_df, args.year, args.magnetisation)

    # Have to convert to 64-bit floats for the amplitude models to evaluate
    # everything correctly, since they convert numpy arrays to
    # C-arrays
    mc_k, mc_pi1, mc_pi2, mc_pi3 = (
        a.astype(np.float64) for a in efficiency_util.k_3pi(pgun_df)
    )
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)

    plotting.z_scatter(
        ag_k,
        ag_pi1,
        ag_pi2,
        ag_pi3,
        mc_k,
        mc_pi1,
        mc_pi2,
        mc_pi3,
        weights,
        mc_corr_wt,
        5,
    )

    fit_suffix = "_fit" if args.fit else ""
    plt.savefig(
        f"z_{args.year}_{args.magnetisation}_data_{args.decay_type}_{args.data_k_charge}"
        f"_weighter_{args.weighter_type}_{args.weighter_k_charge}{fit_suffix}.png"
    )


if __name__ == "__main__":
    parser = common.parser(
        "Using the AmpGen models, plot the measured coherence factor before"
        "and after the reweighting. Splits the data into chunks for better visualisation."
    )
    main(parser.parse_args())
