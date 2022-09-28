"""
Make plots of the variables that will be used for the reweighting

This script doesn't require a reweighter to exist - it's intended to be used to check that you've
created the dataframes correctly.

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import util, get
from lib_efficiency import efficiency_util, plotting


def main(sign: str):
    """
    Create a plot

    """
    ampgen_df = get.ampgen(sign)
    pgun_df = get.particle_gun(sign, show_progress=True)
    # TODO change this to the right thing
    pgun_df = efficiency_util.efficiency_df(pgun_df[pgun_df["train"]])

    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack(
        (helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ampgen_df["time"])
    )
    mc = np.column_stack(
        (helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), pgun_df["time"])
    )

    plotting.projections(mc, ag)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, but don't do the reweighting."
    )
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})

    args = parser.parse_args()
    main(args.sign)
