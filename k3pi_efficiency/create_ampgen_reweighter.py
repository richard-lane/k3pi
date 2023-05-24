"""
Create a reweighter to correct for the mock efficiency
efficiency introduced to the AmpGen dataframes
with the k3pi-data/scripts/add_ampgen_efficiency.py script

"""
import os
import sys
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param
from lib_efficiency import efficiency_definitions, efficiency_util, plotting
from lib_efficiency.reweighter import EfficiencyWeighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))

from lib_data import util, get


def _points(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Phsp points used in parameterisation

    """
    # Get the right arrays
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)

    # Momentum order
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    # Parameterise points
    points = np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))

    return points


def main(args: argparse.Namespace):
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    reweighter_path = efficiency_definitions.ampgen_reweighter_path(args.sign)

    if os.path.exists(reweighter_path):
        raise FileExistsError(reweighter_path)

    if not efficiency_definitions.REWEIGHTER_DIR.is_dir():
        os.mkdir(efficiency_definitions.REWEIGHTER_DIR)

    dataframe = get.ampgen(args.sign)
    dataframe = dataframe[dataframe["train"]]

    target_pts = _points(dataframe)
    original_pts = _points(dataframe[dataframe["accepted"]])

    # Just to check stuff let's plot some projections
    plotting.projections(original_pts, target_pts)
    plt.savefig(f"ampgen_training_proj_{args.sign}.png")
    print("saved fig")

    # Create + train reweighter
    train_kwargs = {
        "n_estimators": 35,  # 350,
        "max_depth": 3,
        # "learning_rate": 0.7,
        "min_samples_leaf": 1800,
    }
    reweighter = EfficiencyWeighter(
        target_pts,
        original_pts,
        np.ones(len(original_pts)),
        fit=False,  # Don't do a time fit (in principle we could)
        min_t=efficiency_definitions.MIN_TIME,
        n_bins=50000,
        n_neighs=30.0,
        **train_kwargs,
    )

    # Dump it
    with open(reweighter_path, "wb") as f:
        print("dumping ampgen reweighter")
        pickle.dump(reweighter, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create efficiency reweighters")
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})

    main(parser.parse_args())
