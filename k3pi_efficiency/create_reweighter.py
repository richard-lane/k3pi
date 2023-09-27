"""
Create a reweighter

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
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_signal_cuts"))

from lib_data import util, d0_mc_corrections
from lib_cuts.get import classifier as get_clf, signal_cut_df
from lib_cuts.definitions import THRESHOLD


def _points(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Phsp points used in parameterisation

    """
    # Get the right arrays
    k, pi1, pi2, pi3 = util.k_3pi(dataframe)

    # Momentum order
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    # Parameterise points
    points = np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))

    return points


def main(
    *, year: str, sign: str, magnetisation: str, k_sign: str, fit: bool, cut: bool
):
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    reweighter_path = efficiency_definitions.reweighter_path(
        year, sign, magnetisation, k_sign, fit, cut
    )
    if os.path.exists(reweighter_path):
        raise FileExistsError(reweighter_path)

    if not efficiency_definitions.REWEIGHTER_DIR.is_dir():
        os.mkdir(efficiency_definitions.REWEIGHTER_DIR)

    # AmpGen points
    ag_pts = _points(efficiency_util.ampgen_df(sign, k_sign, train=True))

    # pgun points
    pgun_df = efficiency_util.pgun_df(year, magnetisation, sign, k_sign, train=True)
    if cut:
        # Always use the DCS BDT for doing cuts
        pgun_df = signal_cut_df(pgun_df, get_clf(year, "dcs", magnetisation), THRESHOLD)
    mc_pts = _points(pgun_df)

    # Do the BDT cut
    # Find D0 MC correction weights
    mc_corr_wts = d0_mc_corrections.pgun_wt_df(pgun_df, year, magnetisation)

    # Just to check stuff let's plot some projections
    fig, axes = plotting.projections(mc_pts, ag_pts, mc_corr_wts)
    suffix = f"{'_fit' if fit else ''}{'_cut' if cut else ''}"
    path = f"training_proj_{year}_{sign}_{magnetisation}_{k_sign}{suffix}.png"

    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)

    plt.savefig(path)
    print(f"saved {path}")

    # Create + train reweighter
    train_kwargs = {
        "n_estimators": 10,  # 600, commented out for pgun mixnig sim thing
        "max_depth": 3,
        "learning_rate": 0.15,
        "min_samples_leaf": 1800,
    }
    reweighter = EfficiencyWeighter(
        ag_pts,
        mc_pts,
        mc_corr_wts,
        fit=fit,
        min_t=efficiency_definitions.MIN_TIME,
        **train_kwargs,
    )

    # Dump it
    with open(reweighter_path, "wb") as f:
        pickle.dump(reweighter, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create efficiency reweighters")
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})
    parser.add_argument("year", type=str, choices={"2017", "2018"})
    parser.add_argument("magnetisation", type=str, choices={"magup", "magdown"})
    parser.add_argument(
        "k_sign",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="Whether to create a reweighter for K+ or K- type evts",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="""Whether to perform a fit to decay times;
                otherwise finds decay time efficiency with a histogram division""",
    )
    parser.add_argument(
        "--cut", action="store_true", help="Whether to perform BDT cut to data"
    )

    main(**vars(parser.parse_args()))
