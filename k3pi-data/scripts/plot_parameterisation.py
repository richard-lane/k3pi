"""
Make plots of phase space variables to make sure everything looks like we expect it to

"""
import sys
import pathlib
import argparse
import resource
from typing import List, Tuple
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_data import definitions, get, util
from lib_efficiency.plotting import phsp_labels


def _plot(
    points: List[np.ndarray], labels: List[str], path: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of points and labels on an axis; return the figure and axis

    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for i, (axis, axis_label) in tqdm(enumerate(zip(axes.ravel(), phsp_labels()))):
        for arr, label in zip(points, labels):
            if label.startswith("False"):
                hist_kw["histtype"] = "stepfilled"
                hist_kw["alpha"] = 0.5

            # Might want to plot up to some maximum lifetime, to illustrate something
            max_lifetimes = 10.0
            mask = arr[:, -1] < max_lifetimes

            # Set the bins by finding automatic bin limits for the first set of points
            if "bins" not in hist_kw:
                _, bins, _ = axis.hist(
                    arr[:, i][mask], bins=100, **hist_kw, label=label
                )
                hist_kw["bins"] = bins

            else:
                axis.hist(arr[:, i], **hist_kw, label=label)
            hist_kw["histtype"] = "step"
            hist_kw["alpha"] = 1

            axis.set_xlabel(axis_label)

        # Remove the bins from the dict once we've plotted all the points
        hist_kw.pop("bins")

    axes.ravel()[-1].legend()

    fig.tight_layout()

    print(f"plotting {path}")
    fig.savefig(path)


def _parameterise(data_frame: pd.DataFrame):
    """
    Find parameterisation of a dataframe

    """
    k = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[0:4]))
    pi1 = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[4:8]))
    pi2 = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[8:12]))
    pi3 = np.row_stack(
        tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[12:16])
    )

    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    return np.column_stack((helicity_param(k, pi1, pi2, pi3), data_frame["time"]))


def _max_usage() -> int:
    """max memory usage of this process in GB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000


def main(*, year: str, magnetisation: str):
    """
    Create a plot

    """
    # These return generators
    n_dfs = 5
    rs_data = _parameterise(
        util.flip_momenta(pd.concat(islice(get.data(year, "cf", magnetisation), n_dfs)))
    )
    print(f"got rs data; {_max_usage()}GB")
    ws_data = _parameterise(
        util.flip_momenta(
            pd.concat(islice(get.data(year, "dcs", magnetisation), n_dfs))
        )
    )
    print(f"got ws data; {_max_usage()}GB")

    rs_pgun = _parameterise(
        util.flip_momenta(
            get.particle_gun(year, "cf", magnetisation, show_progress=True)
        )
    )
    print(f"got rs pgun; {_max_usage()}GB")
    ws_pgun = _parameterise(
        util.flip_momenta(
            get.particle_gun(year, "dcs", magnetisation, show_progress=True)
        )
    )
    print(f"got ws pgun; {_max_usage()}GB")

    # false_df = get.false_sign(show_progress=True)
    # false_sign = _parameterise(
    #     util.flip_momenta(false_df, false_df["K ID"].to_numpy() > 0)
    # )

    rs_mc = _parameterise(util.flip_momenta(get.mc(year, "cf", magnetisation)))
    ws_mc = _parameterise(util.flip_momenta(get.mc(year, "dcs", magnetisation)))

    n_pts = 1_000_000
    rs_ampgen = _parameterise(get.ampgen("cf")[:n_pts])
    ws_ampgen = _parameterise(get.ampgen("dcs")[:n_pts])

    _plot(
        [
            rs_data,
            rs_mc,
            rs_pgun,
            rs_ampgen,
            # false_sign,
        ],
        [
            "CF data",
            "CF MC",
            "CF pgun",
            "CF AmpGen",
            # "False sign pgun",
        ],
        "data_param_rs.png",
    )
    _plot(
        [
            ws_data,
            ws_mc,
            ws_pgun,
            ws_ampgen,
            # false_sign,
        ],
        [
            "DCS data",
            "DCS MC",
            "DCS pgun",
            "DCS AmpGen",
            # "False sign pgun",
        ],
        "data_param_ws.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(**vars(parser.parse_args()))
