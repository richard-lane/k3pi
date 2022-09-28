"""
Train reweighters with various hyper parameters - for each, find the
 - sum of Ks histograms for each projection
 - distance between Z scatter centroids
 - time ratio flatness

and write these to a text file

"""
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from fourbody.param import helicity_param

from lib_data import util
from lib_efficiency import (
    efficiency_util,
    reweighter,
    plotting,
    efficiency_definitions,
)


def _params() -> dict:
    """
    Generates a random set of BDT hyperparameters to use

    """
    gen = np.random.default_rng(
        (os.getpid() * int(time.time()) * int(os.environ["CONDOR_JOB_ID"])) % 123456789
    )
    return {
        "n_estimators": gen.integers(100, 1200),
        "max_depth": gen.integers(2, 9),
        "learning_rate": gen.random(),
        "min_samples_leaf": gen.integers(10, 3000),
    }


def _train(
    ag_df: pd.DataFrame, pgun_df: pd.DataFrame, fit: bool, **train_kw
) -> reweighter.EfficiencyWeighter:
    """
    Train a reweighter

    """
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ag_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = (
        a.astype(np.float64) for a in efficiency_util.k_3pi(pgun_df)
    )

    # Momentum order
    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    # Parameterise points
    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_df["time"]))
    mc = np.column_stack(
        (helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), pgun_df["time"])
    )

    return reweighter.EfficiencyWeighter(
        ag, mc, fit, efficiency_definitions.MIN_TIME, **train_kw
    )


def _weights(
    dataframe: pd.DataFrame, weighter: reweighter.EfficiencyWeighter
) -> np.ndarray:
    """
    Get weights

    """
    k, pi1, pi2, pi3 = (a.astype(np.float64) for a in efficiency_util.k_3pi(dataframe))

    # Momentum order
    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    # Parameterise points
    points = np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))

    return weighter.weights(points)


def _optimise(ampgen_df: pd.DataFrame, pgun_df: pd.DataFrame, fit: bool):
    """
    Run the optimisation, write to file

    """
    # Train a reweighter
    n_repeats = 24
    for _ in range(n_repeats):
        params = _params()
        weighter = _train(
            ampgen_df.loc[ampgen_df["train"]],
            pgun_df.loc[pgun_df["train"]],
            fit=fit,
            **params,
        )

        # Use this weighter to find testing weights
        weights = _weights(pgun_df.loc[~pgun_df["train"]], weighter)

        # TODO Find Ks statistic
        # TODO Find flatness of the time ratio
        # Find distance between Z hulls
        _, _, distance, area = plotting.z_scatter(
            *efficiency_util.k_3pi(ampgen_df.loc[~ampgen_df["train"]]),
            *(
                a.astype(np.float64)
                for a in efficiency_util.k_3pi(pgun_df.loc[~pgun_df["train"]])
            ),
            weights,
            8,
        )

        # Get the parameters used as a tab separated string
        kw_str = str(params).replace("': ", "\t").replace(", '", "\t")[2:-1]

        # Print results
        print(f"{kw_str}\t{distance}\t{area}")


def main(
    sign: str,
    k_sign: str,
    fit: bool,
):
    """
    Run the optimisation

    """
    # Read AmpGen and the right particle gun
    with open(f"ampgen_dataframe_{sign}.pkl", "rb") as f:
        ampgen_df = pickle.load(f)
    with open(f"pgun_dataframe_{sign}.pkl", "rb") as f:
        pgun_df = pickle.load(f)

    # Print options
    print(f"\t".join(f"{x=}" for x in (sign, k_sign, fit)))

    # Deal with getting rid of evts/flipping momenta if we need to
    pgun_df = efficiency_util.k_sign_cut(pgun_df, k_sign)
    if k_sign == "k_minus":
        ampgen_df = util.flip_momenta(
            ampgen_df, np.ones(len(ampgen_df), dtype=np.bool_)
        )

    _optimise(ampgen_df, pgun_df, fit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, and do the reweighting."
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="type of decay, CF or DCS",
    )
    parser.add_argument(
        "k_sign",
        type=str,
        choices={"k_plus", "k_minus"},
        help="whether to read K+ or K- data",
    )
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()
    main(
        args.sign,
        args.k_sign,
        args.fit,
    )
