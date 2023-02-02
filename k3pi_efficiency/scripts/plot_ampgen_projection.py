"""
Make plots of the variables used in the reweighting,
before and after the reweighting for the AmpGen dataframes
with mock efficiency

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import util, get
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_definitions,
    get as eff_get,
)


def main(args):
    """
    Create a plot

    """
    dataframe = get.ampgen(args.sign)
    target_df = dataframe[~dataframe["train"]]
    accepted_df = dataframe[dataframe["accepted"]]

    # Just pass the arrays into the efficiency function and it should find the right weights
    target_k, target_pi1, target_pi2, target_pi3 = efficiency_util.k_3pi(target_df)
    accepted_k, accepted_pi1, accepted_pi2, accepted_pi3 = efficiency_util.k_3pi(
        accepted_df
    )

    # For plotting and finding weights we will want to momentum order
    target_pi1, target_pi2 = util.momentum_order(target_k, target_pi1, target_pi2)
    accepted_pi1, accepted_pi2 = util.momentum_order(
        accepted_k, accepted_pi1, accepted_pi2
    )

    target_t, accepted_t = target_df["time"], accepted_df["time"]

    target_pts = np.column_stack(
        (helicity_param(target_k, target_pi1, target_pi2, target_pi3), target_t)
    )
    accepted_pts = np.column_stack(
        (
            helicity_param(accepted_k, accepted_pi1, accepted_pi2, accepted_pi3),
            accepted_t,
        )
    )

    reweighter = eff_get.ampgen_reweighter_dump(args.sign, verbose=True)
    accepted_wt = reweighter.weights(accepted_pts)
    accepted_wt[accepted_wt > 100] = 0

    # Only keep ampgen events above the mean time so that the plots are scaled the same
    target_pts = target_pts[target_pts[:, -1] > efficiency_definitions.MIN_TIME]

    plotting.projections(accepted_pts, target_pts, mc_wt=accepted_wt)

    plt.savefig(f"ampgen_proj_{args.sign}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot projections of AmpGen's phase space variables"
    )

    parser.add_argument("sign", choices={"dcs", "cf"}, help="Decay type")
    main(parser.parse_args())
