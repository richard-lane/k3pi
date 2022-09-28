"""
Create a reweighter

"""
import os
import sys
import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param
from lib_efficiency import efficiency_definitions, efficiency_util, plotting
from lib_efficiency.reweighter import EfficiencyWeighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))

from lib_data import util


def main(year: str, sign: str, magnetisation: str, k_sign: str, fit: bool):
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    if not efficiency_definitions.REWEIGHTER_DIR.is_dir():
        os.mkdir(efficiency_definitions.REWEIGHTER_DIR)

    # Read the right stuff
    ag_df = efficiency_util.ampgen_df(sign, k_sign, train=True)
    pgun_df = efficiency_util.pgun_df(sign, k_sign, train=True)

    # Get the right arrays
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ag_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    # Momentum order
    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    # Parameterise points
    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_df["time"]))
    mc = np.column_stack(
        (helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), pgun_df["time"])
    )

    # Just to check stuff let's plot some projections
    plotting.projections(mc, ag)
    fit_suffix = "_fit" if fit else ""
    plt.savefig(f"training_proj_{year}_{sign}_{magnetisation}_{k_sign}{fit_suffix}.png")
    print("saved fig")

    # Create + train reweighter
    train_kwargs = {
        "n_estimators": 50,
        "max_depth": 5,
        "learning_rate": 0.7,
        "min_samples_leaf": 1800,
    }
    reweighter = EfficiencyWeighter(
        ag, mc, fit, efficiency_definitions.MIN_TIME, **train_kwargs
    )

    # Dump it
    with open(
        efficiency_definitions.reweighter_path(year, sign, magnetisation, k_sign, fit),
        "wb",
    ) as f:
        pickle.dump(reweighter, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create efficiency reweighters")
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})
    parser.add_argument(
        "k_sign",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="Whether to create a reweighter for K+ or K- type evts",
    )
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation, args.k_sign, args.fit)
