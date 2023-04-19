"""
Plot the angle betwen hadrons from 1 real data file on lxplus

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import sys
import pathlib
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, util, cuts


def _angles(tree, prefix_1: str, prefix_2: str, keep: np.ndarray) -> np.ndarray:
    """
    Angles between hadrons

    """
    return util.relative_angle_branches(tree, prefix_1, prefix_2)[keep]


def _plot(axis: plt.Axes, angles: np.ndarray, label: str) -> None:
    """
    Plot histogram on an axis

    """
    # Colours to use for plotting
    colours = dict(zip(definitions.DATA_BRANCH_PREFICES, ["k", "r", "g", "b", "brown"]))

    hist_kw = {
        "histtype": "step",
        "color": colours[label],
        "label": f"{label}-$\pi_s$ angle",
        "bins": np.linspace(0.0, 0.2, 100),
    }

    axis.hist(angles, **hist_kw)


def main(*, year: str, sign: str, magnetisation: str) -> None:
    """
    Plot the angle between final state hadrons in real data

    """
    # Just use the first few files
    n_files = 5
    data_paths = definitions.data_files(year, magnetisation)[:n_files]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    angles = [[], [], [], []]
    with tqdm(total=n_files * 4) as pbar:
        for path in data_paths:
            with uproot.open(path) as data_f:
                # Find which events to keep (after cuts)
                tree = data_f[definitions.data_tree(sign)]
                keep = (
                    cuts._trigger_keep(tree)
                    & cuts._pid_keep(tree)
                    & cuts._cands_keep(tree)
                    & cuts._sanity_keep(tree)
                )

                for prefix, lst in zip(definitions.DATA_BRANCH_PREFICES[:-1], angles):
                    # Get the angle with the slow pi, plot it in the right colour on the right axis
                    lst.append(
                        _angles(
                            tree,
                            prefix,
                            definitions.DATA_BRANCH_PREFICES[-1],
                            keep,
                        )
                    )
                    pbar.update(1)

    for axis, angle, title in zip(
        axes.ravel(), angles, definitions.DATA_BRANCH_PREFICES[:-1]
    ):
        _plot(axis, np.concatenate(angle), definitions.DATA_BRANCH_PREFICES[-1])
        axis.axvline(cuts.ANGLE_CUT_DEG, color="r", alpha=0.8)
        axis.set_title(title)
        axis.legend()

    fig.tight_layout()
    path = f"data_angle_{year}_{sign}_{magnetisation}.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the angles between hadrons from real data (no cuts)"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2018", "2017"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
