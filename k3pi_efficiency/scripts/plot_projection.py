"""
Make plots of the variables used in the reweighting, before and after the reweighting

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_data import util
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)


def main(args):
    """
    Create a plot

    """
    pgun_df = efficiency_util.pgun_df(args.decay_type, args.data_k_charge, train=False)
    ampgen_df = efficiency_util.ampgen_df(
        args.decay_type, args.data_k_charge, train=False
    )

    # Just pass the arrays into the efficiency function and it should find the right weights
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    ag_t, mc_t = ampgen_df["time"], pgun_df["time"]

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
        verbose=True,
    )

    # For plotting we will want to momentum order
    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_t))
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_t))

    # Only keep ampgen events above the mean time so that the plots are scaled the same
    ag = ag[ag[:, -1] > efficiency_definitions.MIN_TIME]

    plotting.projections(mc, ag, mc_wt=weights)

    fit_suffix = "_fit" if args.fit else ""
    plt.savefig(
        f"proj_{args.year}_{args.magnetisation}_data_{args.decay_type}_{args.data_k_charge}"
        f"_weighter_{args.weighter_type}_{args.weighter_k_charge}{fit_suffix}.png"
    )

    plt.show()


if __name__ == "__main__":
    parser = common.parser("Plot projections of phase space variables")
    main(parser.parse_args())
