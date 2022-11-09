"""
Make plots of the variables used in the reweighting, before the reweighting
With and without BDT cuts

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

from lib_data import util
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_definitions,
    cut,
)


def main(args):
    """
    Create a plot

    """
    pgun_df = efficiency_util.pgun_df(args.sign, k_charge="both", train=False)
    ampgen_df = efficiency_util.ampgen_df(args.sign, k_charge="both", train=False)

    # Find which events we want to keep from the BDT cut
    bdt_keep = cut.mask(pgun_df, args.year, args.magnetisation, args.sign)

    # Parameterise
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    ag_t, mc_t = ampgen_df["time"], pgun_df["time"]

    # For plotting we will want to momentum order
    ag_pi1, ag_pi2 = util.momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = util.momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_t))
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_t))

    # Only keep events above the mean time so that the plots are scaled the same
    ag = ag[ag[:, -1] > efficiency_definitions.MIN_TIME]
    bdt_keep = bdt_keep[mc[:, -1] > efficiency_definitions.MIN_TIME]
    mc = mc[mc[:, -1] > efficiency_definitions.MIN_TIME]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    hist_kw = {"histtype": "step", "density": True}
    for axis, ag_x, mc_x, label in zip(
        axes.ravel(), ag.T, mc.T, plotting.phsp_labels()
    ):
        contents, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC: Before BDT Cut", **hist_kw, alpha=0.5)
        axis.hist(mc_x[bdt_keep], bins=bins, label="MC: After BDT Cut", **hist_kw)
        axis.set_ylim(0, np.max(contents) * 1.1)

        axis.set_xlabel(label)

    axes.ravel()[-1].legend()
    plt.savefig(f"proj_bdtcut_{args.year}_{args.magnetisation}_data_{args.sign}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", choices={"2018"})
    parser.add_argument("magnetisation", choices={"magdown"})
    parser.add_argument("sign", choices={"dcs", "cf"})
    main(parser.parse_args())
