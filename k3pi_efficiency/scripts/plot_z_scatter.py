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

import common
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
)


def main(args: argparse.Namespace):
    """
    Create a plot

    """
    pgun_df = efficiency_util.pgun_df(args.decay_type, args.data_k_charge, train=False)
    ampgen_df = efficiency_util.ampgen_df(
        args.decay_type, args.data_k_charge, train=False
    )

    # Just pass the arrays into the efficiency function and it should find the right weights
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    # Have to convert to 64-bit floats for the amplitude models to evaluate
    # everything correctly, since they convert numpy arrays to
    # C-arrays
    mc_k, mc_pi1, mc_pi2, mc_pi3 = (
        a.astype(np.float64) for a in efficiency_util.k_3pi(pgun_df)
    )

    mc_t = pgun_df["time"]

    weights = efficiency_model.weights(
        mc_k,
        mc_pi1,
        mc_pi2,
        mc_pi3,
        mc_t,
        args.weighter_k_charge,
        args.year,
        args.weighter_type,
        args.magnetisation,
        args.fit,
        args.cut,
        verbose=True,
    )

    plotting.z_scatter(
        ag_k, ag_pi1, ag_pi2, ag_pi3, mc_k, mc_pi1, mc_pi2, mc_pi3, weights, 5
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
