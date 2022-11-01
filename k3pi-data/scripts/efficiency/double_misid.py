"""
Measure how often we confuse the slow pion for one
of the K3pi daughters in MC.

Do this by comparing the true origin vertices of
the daughters actually match the end vertices of the
D0 or D*.

"""
import sys
import pathlib
from typing import Tuple, List
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from lib_data import util, definitions, cuts
from lib_efficiency.efficiency_util import k_3pi


def _binomial_err(n_tot: int, n_success: int) -> float:
    """Find std deviation on the number of successes"""
    p = n_success / n_tot
    return np.sqrt(n_tot * p * (1 - p))


def _keep(tree) -> np.ndarray:
    # Commented out - meant we never got any swaps.
    # might not be the right thing to do
    # return (np.abs(tree["Dst_TRUEID"].array(library="np")) == 413) & (
    #     np.abs(tree["D0_TRUEID"].array(library="np")) == 421
    # )

    # Keep only +ve times
    return (
        tree["Dst_ReFit_D0_ctau"].array(library="ak")[:, 0] > 0
    ) & cuts._sanity_keep(tree)


def _position(tree, branch_prefix: str) -> np.ndarray:
    """
    x and y position as a (N, 2) shape array

    """
    return np.column_stack(
        [tree[f"{branch_prefix}{x}"].array(library="np") for x in ("X", "Y")]
    )


def _close(posn1: np.ndarray, posn2: np.ndarray) -> np.ndarray:
    """
    Bool array of whether two positions are close

    """
    return np.isclose(posn1, posn2).all(axis=1)


def _swapped_daughters(tree) -> List:
    """
    Named tuple of boolean masks telling us whether each daughter
    particle has been doubly-misID'd for the slow pion

    Uses the X position of the end/origin vertex to determine
    whether we've misID'd stuff

    :returns: namedtuple of arrays whether each daughter has been swapped with the slow pi

    """
    # Find the D0 end vertices
    d0_end = _position(tree, "D0_TRUEENDVERTEX_")

    # Find the D* end vertices
    dst_end = _position(tree, "Dst_TRUEENDVERTEX_")

    # Decay products origin vertices
    slowpi_origin = _position(tree, "Dst_pi_TRUEORIGINVERTEX_")

    # indices in this order to get the particle ordering right
    daughters_origin = [
        _position(tree, f"D0_P{i}_TRUEORIGINVERTEX_") for i in (0, 1, 3, 2)
    ]

    # Find whether we've swapped stuff
    slowpi_swapped = _close(slowpi_origin, d0_end)
    daughters_swapped = [
        _close(daughter_origin, dst_end) for daughter_origin in daughters_origin
    ]

    # Doubly mis-IDd if both the daughter and slow pi are misIDd
    return [slowpi_swapped & arr for arr in daughters_swapped]


def _indices(tree) -> np.ndarray:
    """
    Bin indices for K3pi daughters based on the helicity parameterisation

    :returns: array of values {0, 1, 2, 3} for the phsp bin

    """
    # Read momenta
    # The easiest way to do this is to create a dataframe
    dataframe = pd.DataFrame()
    util.add_momenta(dataframe, tree, np.s_[:])  # no cuts yet

    # Phase space parameterisation
    points = helicity_param(*k_3pi(dataframe))

    # Divide into +-ve phi (4), costheta+ (2) > or < mean
    # 4 bins based on (2 * phi bin + cos theta bin)
    theta_boundary, phi_boundary = (0.0, 0.0)

    theta_bin = (points[:, -2] < theta_boundary).astype(int)
    phi_bin = (points[:, -1] < phi_boundary).astype(int)

    return 2 * phi_bin + theta_bin


def measure_swap(year: str, magnetisation: str, sign: str) -> Tuple[np.ndarray, int]:
    """
    Using the origin/end vertex positions, find whether we've
    doubly misID'd something

    Then plot phase space averaged and phase space binned
    histograms of misID rate

    returns 2d array:
    [[bin1: K, pi1, pi2, pi3], [bin2: K, pi1, pi2, pi3] ...] double misID probs

    """
    n_tot = 0

    n_swap = [[0 for _ in range(4)] for _ in range(4)]
    for data_path in tqdm(definitions.mc_files(year, magnetisation, sign)):
        with uproot.open(data_path) as data_f:
            tree = data_f[definitions.data_tree(sign)]

            # We may only want to keep some of our events
            keep = _keep(tree)

            # Measure whether each daughter particle has been swapped with the slow pion
            swaps = [arr[keep] for arr in _swapped_daughters(tree)]

            # Bin the phase space into four regions based on the helicity parameterisation
            bin_indices = _indices(tree)[keep]

            # Count how many in each bin are/aren't swapped
            n_tot += len(bin_indices)
            for i, swap in enumerate(swaps):
                for j in range(4):
                    this_bin = bin_indices == j
                    n_swap[j][i] += np.sum(swap[this_bin])

    return np.array(n_swap), n_tot


def main():
    year, magnetisation = "2018", "magdown"

    ws_swap, ws_tot = measure_swap(year, magnetisation, "dcs")
    rs_swap, rs_tot = measure_swap(year, magnetisation, "cf")

    # Make plots with binomial error bars of how many RS/WS
    # type were swapped in each phase space bin
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for i, (axis, particle) in enumerate(
        zip(ax.ravel(), ("K", r"$\pi_1$", r"$\pi_2$", r"$\pi_3$"))
    ):
        ws_success = ws_swap[:, i]
        rs_success = rs_swap[:, i]

        ws_rate = ws_success / ws_tot
        rs_rate = rs_success / rs_tot

        ws_err = _binomial_err(ws_tot, ws_success) / ws_tot
        rs_err = _binomial_err(rs_tot, rs_success) / rs_tot

        # Convert to %
        ws_rate *= 100
        rs_rate *= 100
        ws_err *= 100
        rs_err *= 100

        axis.errorbar(
            range(4),
            ws_rate,
            yerr=ws_err,
            fmt="+",
            label=r"WS; $D^{*+}\rightarrow (K^+\pi_1^-\pi_2^-\pi_3^+)\pi_s^+$",
        )
        axis.errorbar(
            range(4),
            rs_rate,
            yerr=rs_err,
            fmt="+",
            label=r"RS; $D^{*+}\rightarrow (K^-\pi_1^+\pi_2^+\pi_3^-)\pi_s^+$",
        )
        axis.set_title(particle)

        axis.set_xlabel(r"Bin Number")
        axis.set_ylabel(r"MisID rate /%")

        axis.set_xticks(range(4))
        axis.set_ylim(0.0, 0.11)

    ax[0, 0].legend()

    fig.tight_layout()
    fig.savefig("double_misid.png")
    plt.show()


if __name__ == "__main__":
    main()
