"""
Plot the angle betwen hadrons from 1 real data file on lxplus

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import sys
import pathlib
import argparse

import matplotlib.pyplot as plt
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, util


def _plot(axis: plt.Axes, prefix_1: str, prefix_2: str) -> None:
    """
    Plot histogram on an axis

    """
    axis.set_title(prefix_1)

    # Colours to use for plotting
    colours = dict(zip(definitions.DATA_BRANCH_PREFICES, ["k", "r", "g", "b", "brown"]))

    hist_kw = {"histtype": "step", "color": colours[prefix_2], "label": prefix_2}

    angles = util.relative_angle_branches(tree, prefix_1, prefix_2)

    axis.hist(angles, **hist_kw)


def main(*, year: str, sign: str, magnetisation: str) -> None:
    """
    Plot the angle between final state hadrons in real data

    """
    # Just use the first file
    data_path = definitions.data_files(year, magnetisation)[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    with uproot.open(data_path) as data_f:
        for prefix, axis in zip(definitions.DATA_BRANCH_PREFICES[:-1], axes.ravel()):
            # Get the angle, plot it in the right colour on the right axis
            for target_prefix in definitions.DATA_BRANCH_PREFICES:
                if target_prefix == prefix:
                    continue

                _plot(axis, prefix_1, prefix_2)

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
        choices={"2018"},
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
        choices={"magdown"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
