"""
Library functions and such for doing the D0
momentum MC corrections

"""
import pickle
import pathlib
from typing import Tuple, Union, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from hep_ml.reweight import BinsReweighter


class EtaPWeighter:
    """
    For reweighting the D0 Eta/P distributions

    Also for plotting, which is technically a different
    responsibility but I don't really mind

    """

    def __init__(
        self,
        target: np.ndarray,
        original: np.ndarray,
        target_wt: np.ndarray = None,
        original_wt: np.ndarray = None,
        **weighter_kw,
    ):
        """
        Build histograms

        :param target: (2, N) shape array of (eta, P)
        :param original: (2, N) shape array of (eta, P)

        """
        self._target = target
        self._orig = original

        self._target_wt = target_wt
        self._orig_wt = original_wt

        self._reweighter = BinsReweighter(**weighter_kw)
        self._reweighter.fit(
            original.T, target.T, original_weight=original_wt, target_weight=target_wt
        )

    def weights(self, points: np.ndarray, wts: np.ndarray = None) -> np.ndarray:
        """
        Weights for an array of points

        :param points: (2, N) shape array of (eta, P)

        """
        return self._reweighter.predict_weights(points.T, original_weight=wts)

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
        weights: np.ndarray,
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

        eta_axis.hist(
            eta, **eta_kw, **hist_kw, orientation="horizontal", weights=weights
        )

        # Manually scale y axis otherwise it breaks
        counts, _, _ = p_axis.hist(momm, **p_kw, **hist_kw, weights=weights)
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
        points: np.ndarray,
        weights: np.ndarray,
        bins: Tuple[np.ndarray, np.ndarray],
        path: str,
    ) -> None:
        """
        Make 2d + 1d plots of eta/phi

        :param points: (2, N) array of eta, phi
        :param path: where to save to

        """
        eta, momm = points
        fig, axes = EtaPWeighter._fig_ax()

        # 2d plot
        eta_bins, p_bins = bins

        _, _, _, image = axes["A"].hist2d(
            momm, eta, bins=(p_bins, eta_bins), norm=LogNorm(), weights=weights
        )

        EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            (eta, momm),
            weights,
            eta_kw={"bins": eta_bins},
            p_kw={"bins": p_bins},
        )

        fig.colorbar(image, cax=axes["B"])

        fig.suptitle(path)
        EtaPWeighter._format(fig, axes)

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
        weights = self._target_wt if distribution == "target" else self._orig_wt

        EtaPWeighter._plot(data, weights, bins, path)

    def plot_ratio(self, bins: Tuple[np.ndarray, np.ndarray], path: str) -> None:
        """
        Plot the ratio of two distributions and the 1d projections of ratio

        """
        fig, axes = EtaPWeighter._fig_ax()
        eta_bins, p_bins = bins

        # 2d plot
        target_count_2d, _, _ = np.histogram2d(
            *self._target,
            bins=(eta_bins, p_bins),
            density=True,
            weights=self._target_wt,
        )
        orig_count_2d, _, _ = np.histogram2d(
            *self._orig, bins=(eta_bins, p_bins), density=True, weights=self._orig_wt
        )
        ratio_2d = orig_count_2d / target_count_2d

        extent = [*p_bins[[0, -1]], *eta_bins[[0, -1]]]
        image = axes["A"].matshow(
            ratio_2d, norm=LogNorm(), extent=extent, aspect="auto", origin="lower"
        )

        # 1d hists
        EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            self._target,
            self._target_wt,
            eta_kw={"bins": eta_bins, "density": True},
            p_kw={"bins": p_bins, "label": "Target", "density": True},
        )
        EtaPWeighter._proj_plots(
            (axes["D"], axes["C"]),
            self._orig,
            self._orig_wt,
            eta_kw={"bins": eta_bins, "density": True},
            p_kw={"bins": p_bins, "label": "Original", "density": True},
        )

        fig.colorbar(image, cax=axes["B"])

        fig.suptitle(path)
        axes["C"].legend()
        EtaPWeighter._format(fig, axes)

        print(f"saving {path}")
        plt.savefig(path)
        plt.close(fig)


def d0_points(dfs: Union[Iterable[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
    """
    Eta, phi arrays from either a dataframe or a generator of dataframes

    :param dfs: either a single dataframe or an iterable of dataframes

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


def _dir():
    """
    Directory where the dumps live

    """
    return pathlib.Path(__file__).resolve().parents[1] / "d0_weighters"


def pgun_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Absolute path to a reweighter used for correcting particle gun D0
    eta/momentum distributions to data

    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"

    :returns: path

    """
    assert year in {"2018"}
    assert sign in {"dcs", "cf"}
    assert magnetisation in {"magdown", "magup"}

    return _dir() / f"pgun2data_{year}_{sign}_{magnetisation}.pkl"


def mc_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Absolute path to a reweighter used for correcting MC D0
    eta/momentum distributions to data

    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"

    :returns: path

    """
    assert year in {"2018"}
    assert sign in {"dcs", "cf"}
    assert magnetisation in {"magdown", "magup"}

    return _dir() / f"mc2data_{year}_{sign}_{magnetisation}.pkl"


def get_pgun(year: str, sign: str, magnetisation: str) -> EtaPWeighter:
    """
    Unpickle a reweighter from a dump

    """
    with open(pgun_path, "rb", encoding="utf8"):
        return pickle.load(pgun_path(year, sign, magnetisation))


def get_mc(year: str, sign: str, magnetisation: str) -> EtaPWeighter:
    """
    Unpickle a reweighter from a dump

    """
    with open(pgun_path, "rb", encoding="utf8"):
        return pickle.load(mc_path(year, sign, magnetisation))


def pgun_weights(
    dfs: Union[Iterable[pd.DataFrame], pd.DataFrame],
    year: str,
    sign: str,
    magnetisation: str,
) -> np.ndarray:
    """
    Weights for particle gun -> data

    """
    weighter = get_pgun(year, sign, magnetisation)

    return weighter.weights(d0_points(dfs))


def mc_weights(
    dfs: Union[Iterable[pd.DataFrame], pd.DataFrame],
    year: str,
    sign: str,
    magnetisation: str,
) -> np.ndarray:
    """
    Weights for MC -> data

    """
    weighter = get_mc(year, sign, magnetisation)

    return weighter.weights(d0_points(dfs))
