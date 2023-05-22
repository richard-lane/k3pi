"""
Make plots of the variables used in the reweighting, before and after the reweighting

"""
import sys
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import inv_mass_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_data import util, d0_mc_corrections
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)
from lib_efficiency.get import reweighter_dump

from lib_cuts.get import classifier as get_clf, signal_cut_df
from lib_cuts.definitions import THRESHOLD


def main(
    *,
    year: str,
    decay_type: str,
    weighter_type: str,
    magnetisation: str,
    data_k_charge: str,
    weighter_k_charge: str,
    fit: bool,
    cut: bool,
):
    """
    Create a plot

    """
    pgun_df = efficiency_util.pgun_df(
        year, magnetisation, decay_type, data_k_charge, train=False
    )
    ampgen_df = efficiency_util.ampgen_df(decay_type, data_k_charge, train=False)

    # We might want to do BDT cut
    if cut:
        # Always use DCS classifier for BDT cut, even on RS data
        pgun_df = signal_cut_df(pgun_df, get_clf(year, "dcs", magnetisation), THRESHOLD)

    reweighter = reweighter_dump(
        year,
        weighter_type,
        magnetisation,
        weighter_k_charge,
        fit,
        cut,
        verbose=True,
    )
    weights = efficiency_util.wts_df(pgun_df, reweighter)

    # Get D0 MC corr wts
    mc_corr_wt = d0_mc_corrections.pgun_wt_df(pgun_df, year, magnetisation)

    # Get arrays
    ag_k, ag_pi1, ag_pi2, ag_pi3 = util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = util.k_3pi(pgun_df)

    # Momentum order for the plot
    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    ag_t, mc_t = ampgen_df["time"], pgun_df["time"]

    ag = np.column_stack((inv_mass_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_t))
    mc = np.column_stack((inv_mass_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_t))

    # Only keep ampgen events above the mean time so that the plots are scaled the same
    ag = ag[ag[:, -1] > efficiency_definitions.MIN_TIME]

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    hist_kw = {"density": True, "histtype": "step"}
    labels = (
        r"$M(K+\pi_1^-)$",
        r"$M(\pi_1^-\pi_2^-)$",
        r"$M(\pi_2^-\pi^+)$",
        r"$M(K^+\pi_1^-\pi_2^-)$",
        r"$M(\pi_1^-\pi_2^-\pi^+)$",
        r"t/$\tau$",
    )
    for axis, ag_x, mc_x, label in zip(ax.ravel(), ag.T, mc.T, labels):
        contents, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw, alpha=0.5, weights=mc_corr_wt)
        axis.hist(
            mc_x,
            bins=bins,
            label="Reweighted",
            **hist_kw,
            weights=weights * mc_corr_wt,
        )
        axis.set_ylim(0, np.max(contents) * 1.1)

        axis.set_xlabel(label)

    ax[0, 0].legend()
    fig.tight_layout()
    path = f"alt_proj_{year}_{magnetisation}_data_{decay_type}_{data_k_charge}_weighter_{weighter_type}_{weighter_k_charge}_{fit=}_{cut=}.png"
    fig.savefig(path)

    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, ax), f)


if __name__ == "__main__":
    parser = common.parser("Plot projections of phase space variables")
    main(**vars(parser.parse_args()))
