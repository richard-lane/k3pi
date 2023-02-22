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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get


class D0EtaPWeighter:
    """For reweighting the D0 Eta/P distributions"""

    def __init__(
        self,
        target: np.ndarray,
        original: np.ndarray,
        eta_bins: np.ndarray,
        p_bins: np.ndarray,
    ):
        """
        Build histograms

        :param target: (2, N) shape array of (eta, P)
        :param original: (2, N) shape array of (eta, P)

        """
        self._eta_bins = eta_bins
        self._p_bins = p_bins

        self._target_counts, _, _ = np.histogram2d(
            *target, bins=(eta_bins, p_bins), density=True
        )
        self._orig_counts, _, _ = np.histogram2d(
            *original, bins=(eta_bins, p_bins), density=True
        )

        # Pad with 0s to deal with over/underflow
        self._ratio = np.pad(
            self._target_counts / self._orig_counts, 1, mode="constant"
        )

        # Infs set to nan
        self._ratio[np.isinf(self._ratio)] = np.nan

    def weights(self, points: np.ndarray) -> np.ndarray:
        """
        Weights for an array of points

        """
        eta_indices = np.digitize(points[0], self._eta_bins)
        p_indices = np.digitize(points[1], self._p_bins)

        weights = self._ratio[eta_indices, p_indices]

        return weights

    def plot(self) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the 2d histograms and their ratio

        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        extent = [*self._p_bins[[0, -1]], *self._eta_bins[[0, -1]]]

        for axis, data, title in zip(
            axes,
            (self._orig_counts, self._target_counts, self._ratio),
            ("Target", "Original", "Ratio"),
        ):
            axis.matshow(
                data, norm=LogNorm(), extent=extent, aspect="auto", origin="lower"
            )
            axis.set_title(title)

        return fig, axes


def _plot(dfs: Union[Iterable[pd.DataFrame], pd.DataFrame], path: str) -> None:
    """
    Make 2d + 1d plots of eta/phi

    """
    if isinstance(dfs, pd.DataFrame):
        # MC or pgun - passed a single dataframe
        d0_eta = dfs["D0 eta"]
        d0_p = dfs["D0 P"]

    else:
        # real data - passed a generator
        arrs = [(dataframe["D0 eta"], dataframe["D0 P"]) for dataframe in dfs]

        d0_eta = np.concatenate([arr[0] for arr in arrs])
        d0_p = np.concatenate([arr[1] for arr in arrs])

    fig, axes = plt.subplot_mosaic(
        "DAAAAB\nDAAAAB\nDAAAAB\nDAAAAB\n.CCCC.", figsize=(6, 5)
    )

    # 2d plot
    eta_bins = np.linspace(1.5, 5.5, 30)
    p_bins = np.linspace(0.0, 500000, 30)

    _, _, _, image = axes["A"].hist2d(
        d0_p, d0_eta, bins=(p_bins, eta_bins), norm=LogNorm()
    )

    hist_kw = {"histtype": "step"}
    axes["D"].hist(d0_eta, bins=eta_bins, **hist_kw, orientation="horizontal")

    # Otherwise the yscale breaks
    p_counts, _, _ = axes["C"].hist(d0_p, bins=p_bins, **hist_kw)
    axes["C"].set_ylim(0, 1.05 * np.max(p_counts))

    fig.colorbar(image, cax=axes["B"])

    axes["A"].set_xticks([])
    axes["A"].set_yticks([])

    axes["D"].set_xticks([])
    axes["C"].set_yticks([])

    axes["D"].set_ylabel(r"$\eta$", rotation=0)
    axes["C"].set_xlabel("P")

    fig.suptitle(path)
    fig.tight_layout()

    print(f"saving {path}")
    plt.savefig(path)
    plt.close(fig)


def main(*, year: str, magnetisation: str, sign: str):
    """
    Get particle gun, MC and data dataframes
    Make 1 and 2d histograms of D0 eta and P
    Plot and show them

    """
    pgun = get.particle_gun(sign)
    mc = get.mc(year, sign, magnetisation)

    _plot(pgun, f"d0_eta_p_pgun_{year}_{magnetisation}_{sign}.png")
    _plot(mc, f"d0_eta_p_mc_{year}_{magnetisation}_{sign}.png")
    _plot(
        get.data(year, sign, magnetisation),
        f"d0_eta_p_data_{year}_{magnetisation}_{sign}.png",
    )

    eta_bins = np.linspace(1.5, 5.5, 100)
    p_bins = np.linspace(0.0, 700000, 100)
    mc_pts = np.row_stack([mc[label] for label in ("D0 eta", "D0 P")])
    pgun_pts = np.row_stack([pgun[label] for label in ("D0 eta", "D0 P")])

    # Weight MC -> Pgun as a test
    weighter = D0EtaPWeighter(
        mc_pts,
        pgun_pts,
        eta_bins,
        p_bins,
    )
    weighter.plot()
    plt.show()
    plt.close(plt.gcf())

    weights = weighter.weights(mc_pts)
    weights[np.isnan(weights)] = 0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    kw = {"histtype": "step", "density": True}
    ax[0].hist(mc_pts[0], bins=eta_bins, label="MC", **kw)
    ax[0].hist(pgun_pts[0], bins=eta_bins, label="pgun", **kw)
    ax[0].hist(mc_pts[0], bins=eta_bins, label="weighted", weights=weights, **kw)

    ax[1].hist(mc_pts[1], bins=p_bins, label="MC", **kw)
    ax[1].hist(pgun_pts[1], bins=p_bins, label="pgun", **kw)
    ax[1].hist(mc_pts[1], bins=p_bins, label="weighted", weights=weights, **kw)

    ax[0].legend()

    plt.show()


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
