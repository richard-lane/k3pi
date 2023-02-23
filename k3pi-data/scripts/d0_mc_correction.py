"""
Make histograms of D0 eta and momentum;
show them and save to a dump somewhere

"""
import sys
import pathlib
import argparse
from typing import Tuple, Union, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from hep_ml.reweight import BinsReweighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get


class D0EtaPWeighter:
    """For reweighting the D0 Eta/P distributions"""

    def __init__(
        self,
        target: np.ndarray,
        original: np.ndarray,
        **weighter_kw,
    ):
        """
        Build histograms

        :param target: (2, N) shape array of (eta, P)
        :param original: (2, N) shape array of (eta, P)

        """
        self._target = target
        self._orig = original

        self._reweighter = BinsReweighter(**weighter_kw)
        self._reweighter.fit(original.T, target.T)

    def weights(self, points: np.ndarray) -> np.ndarray:
        """
        Weights for an array of points

        :param points: (2, N) shape array of (eta, P)

        """
        return self._reweighter.predict_weights(points.T)

    @staticmethod
    def _fig_ax() -> Tuple[plt.Figure, np.ndarray]:
        """Figure + axes for 2d plot with projections"""
        return plt.subplot_mosaic(
            "DAAAAB\nDAAAAB\nDAAAAB\nDAAAAB\n.CCCC.", figsize=(6, 5)
        )

    @staticmethod
    def _proj_plots(
        axes: Tuple[plt.Axes, plt.Axes],
        points: Tuple[np.ndarray, np.ndarray],
        *,
        eta_kw: dict = None,
        p_kw: dict = None,
    ) -> None:
        """
        Make projections of eta and p projections on axes

        :param axes: (eta axis, p axis)
        :param points: (eta, p)

        """
        if eta_kw is None:
            eta_kw = {}
        if p_kw is None:
            p_kw = {}

        hist_kw = {"histtype": "step"}

        eta_axis, p_axis = axes
        eta, momm = points

        eta_axis.hist(eta, **eta_kw, **hist_kw, orientation="horizontal")

        # Manually scale y axis otherwise it breaks
        counts, _, _ = p_axis.hist(momm, **p_kw, **hist_kw)
        new_lim = 1.1 * np.max(counts[np.isfinite(counts)])
        p_axis.set_ylim(0.0, new_lim)

    @staticmethod
    def _format(fig: plt.Figure, axes: dict) -> None:
        """
        Format a figure and axes so it looks nice

        """
        axes["A"].set_xticks([])
        axes["A"].set_yticks([])

        axes["D"].set_xticks([])
        axes["C"].set_yticks([])

        axes["D"].set_ylabel(r"$\eta$", rotation=0)
        axes["C"].set_xlabel("P")

        fig.tight_layout()

    @staticmethod
    def _plot(
        points: np.ndarray, bins: Tuple[np.ndarray, np.ndarray], path: str
    ) -> None:
        """
        Make 2d + 1d plots of eta/phi

        :param points: (2, N) array of eta, phi
        :param path: where to save to

        """
        eta, momm = points
        fig, axes = D0EtaPWeighter._fig_ax()

        # 2d plot
        eta_bins, p_bins = bins

        _, _, _, image = axes["A"].hist2d(
            momm, eta, bins=(p_bins, eta_bins), norm=LogNorm()
        )

        D0EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            (eta, momm),
            eta_kw={"bins": eta_bins},
            p_kw={"bins": p_bins},
        )

        fig.colorbar(image, cax=axes["B"])

        fig.suptitle(path)
        D0EtaPWeighter._format(fig, axes)

        print(f"saving {path}")
        plt.savefig(path)
        plt.close(fig)

    def plot_distribution(
        self, distribution: str, bins: Tuple[np.ndarray, np.ndarray], path: str
    ) -> None:
        """
        Plot a 2d histograms and their projections

        :param distribution: "target" or "original"
        :param path: where to save figure

        """
        assert distribution in {"target", "original"}

        data = self._target if distribution == "target" else self._orig

        D0EtaPWeighter._plot(data, bins, path)

    def plot_ratio(self, bins: Tuple[np.ndarray, np.ndarray], path: str) -> None:
        """
        Plot the ratio of two distributions and the 1d projections of ratio

        """
        fig, axes = D0EtaPWeighter._fig_ax()
        eta_bins, p_bins = bins

        # 2d plot
        target_count_2d, _, _ = np.histogram2d(
            *self._target, bins=(eta_bins, p_bins), density=True
        )
        orig_count_2d, _, _ = np.histogram2d(
            *self._orig, bins=(eta_bins, p_bins), density=True
        )
        ratio_2d = orig_count_2d / target_count_2d

        extent = [*p_bins[[0, -1]], *eta_bins[[0, -1]]]
        image = axes["A"].matshow(
            ratio_2d, norm=LogNorm(), extent=extent, aspect="auto", origin="lower"
        )

        # 1d hists
        D0EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            self._target,
            eta_kw={"bins": eta_bins, "density": True},
            p_kw={"bins": p_bins, "label": "Target", "density": True},
        )
        D0EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            self._orig,
            eta_kw={"bins": eta_bins, "density": True},
            p_kw={"bins": p_bins, "label": "Original", "density": True},
        )

        fig.colorbar(image, cax=axes["B"])

        fig.suptitle(path)
        axes["C"].legend()
        D0EtaPWeighter._format(fig, axes)

        print(f"saving {path}")
        plt.savefig(path)
        plt.close(fig)


def _points(dfs: Union[Iterable[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
    """
    Eta, phi arrays from either a dataframe or a generator of dataframes

    """
    if isinstance(dfs, pd.DataFrame):
        # MC or pgun - passed a single dataframe
        d0_eta = dfs["D0 eta"]
        d0_p = dfs["D0 P"]

        retval = np.row_stack((d0_eta, d0_p))

    else:
        # real data - passed a generator
        arrs = [
            tuple(dataframe[label].to_numpy() for label in ("D0 eta", "D0 P"))
            for dataframe in dfs
        ]

        retval = np.concatenate(arrs, axis=1)

    return retval


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get particle gun, MC and data dataframes
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    pgun_pts = _points(get.particle_gun(sign))
    mc_pts = _points(get.mc(year, sign, magnetisation))
    data_pts = _points(get.data(year, sign, magnetisation))

    # Bins for plotting
    # The reweighter doesn't use these bins, it finds its own
    bins = (np.linspace(1.5, 5.5, 100), np.linspace(0.0, 500000, 100))

    # Weight pgun -> data as a test
    weighter = D0EtaPWeighter(
        data_pts,
        pgun_pts,
        n_bins=100,
        n_neighs=0.5,
    )

    weighter.plot_distribution("target", bins, "d0_correction_data.png")
    weighter.plot_distribution("original", bins, "d0_correction_pgun.png")
    weighter.plot_ratio(bins, "d0_correction_data_pgun_ratio.png")

    # Make plots showing the reweighting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    kw = {"histtype": "step", "density": True}
    axes[0].hist(mc_pts[0], bins=bins[0], label="mc", **kw)
    axes[0].hist(data_pts[0], bins=bins[0], label="data", **kw)
    axes[0].hist(
        mc_pts[0],
        bins=bins[0],
        label="weighted",
        weights=weighter.weights(mc_pts),
        **kw,
    )

    axes[1].hist(mc_pts[1], bins=bins[1], label="mc", **kw)
    counts, _, _ = axes[1].hist(data_pts[1], bins=bins[1], label="data", **kw)
    axes[1].hist(
        mc_pts[1],
        bins=bins[1],
        label="weighted",
        weights=weighter.weights(mc_pts),
        **kw,
    )
    axes[1].set_ylim(0.0, 1.2 * np.max(counts))
    axes[0].legend()

    fig.savefig("d0_correction_data_to_pgun.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make + store histograms of D0 eta and P"
    )
    parser.add_argument("year", type=str, help="data taking year", choices={"2018"})
    parser.add_argument(
        "magnetisation", type=str, help="mag direction", choices={"magup", "magdown"}
    )
    parser.add_argument("sign", type=str, help="decay type", choices={"dcs", "cf"})

    main(**vars(parser.parse_args()))
