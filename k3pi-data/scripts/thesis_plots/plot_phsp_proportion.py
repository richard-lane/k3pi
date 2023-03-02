"""
Plot stacked hists showing the proportion of events in each phase space bin,
as functions of time and the phase space dimensions

"""
import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_efficiency"))

from lib_data import get, util
from lib_efficiency.efficiency_util import points
from lib_efficiency.plotting import phsp_labels


def _points(year: str, magnetisation: str, sign: str) -> np.ndarray:
    """
    Get a generator of dataframes, get the points from it

    """
    pts = (
        points(*util.k_3pi(dataframe), dataframe["time"])
        for dataframe in get.data(year, sign, magnetisation)
    )

    return np.concatenate(list(pts))


def _phsp_bins(year: str, magnetisation: str, sign: str) -> np.ndarray:
    """
    Get the phsp bin indices

    """
    return np.concatenate(
        list(dataframe["phsp bin"] for dataframe in get.data(year, sign, magnetisation))
    )


def _plot(pts: np.ndarray, bin_indices: np.ndarray) -> plt.Figure:
    """
    Plot a stacked, normalised histogram showing the proportion of events in each
    phase space bin as a function of phase space variables

    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    hist_bins = [
        np.linspace(min_, max_, 100)
        for (min_, max_) in zip(
            [600, 200, -1.0, -1.0, -np.pi, 0.0], [1600, 1200, 1.0, 1.0, np.pi, 10.0]
        )
    ]

    colours = "r", "g", "b", "y", "k"
    hist_kw = {"histtype": "stepfilled", "color": colours, "stacked": True}
    for point, axis, bins, label in zip(pts.T, axes.ravel(), hist_bins, phsp_labels()):
        plot_kw = {**hist_kw, "bins": bins}
        indices = np.unique(bin_indices)

        axis.hist([point[bin_indices == i] for i in indices], **plot_kw, label=indices)

        axis.set_xlabel(label)

    axes.ravel()[-1].legend()
    fig.tight_layout()

    return fig


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get the right dataframes,
    get points from each,
    concatenate them into a big array,
    plot them as a stacked hist

    """
    fig = _plot(
        _points(year, magnetisation, sign), _phsp_bins(year, magnetisation, sign)
    )

    path = str(
        pathlib.Path(__file__).resolve().parents[0]
        / f"phsp_proportion_{year}_{magnetisation}_{sign}.png"
    )

    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot stacked hists showing the phase space binning and parameterisation"
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
