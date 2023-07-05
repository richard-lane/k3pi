"""
Toy reweighting by applying an analytic efficiency function to the AmpGen training set,
training a reweighter and comparing it to the true distributions.

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param, inv_mass_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))
from lib_data import get, util
from lib_data.plotting import pull
from lib_efficiency import (
    time_fitter,
    plotting,
    efficiency_util,
    efficiency_definitions,
    get as eff_get,
)
from lib_efficiency.reweighter import EfficiencyWeighter
from lib_time_fit.definitions import TIME_BINS


def _time_efficiency(
    times: np.ndarray, time_params: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Decay time efficiency

    """
    n, m, a, b = time_params
    return (
        time_fitter.normalised_pdf(times, 0, n, m, a, b, 1)[1] * np.exp(times) / 3.2
    )  # Scale to make the efficiency < 1


def _pt(p_x: np.ndarray, p_y: np.ndarray) -> np.ndarray:
    return np.sqrt(p_x**2 + p_y**2)


def _pt_efficiency(p_t: np.ndarray, scale: float) -> np.ndarray:
    """
    Efficiency based on pT

    """
    return scale + (1 - scale) * np.sin(p_t / 200)


def _k_efficiency(p_t: np.ndarray) -> np.ndarray:
    return _pt_efficiency(p_t, 0.60)


def _piminus_efficiency(p_t: np.ndarray) -> np.ndarray:
    return _pt_efficiency(p_t, 0.75)


def _piplus_efficiency(p_t: np.ndarray) -> np.ndarray:
    return _pt_efficiency(p_t, 0.90)


def _efficiency(
    dataframe: pd.DataFrame, time_params: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Mask to apply an efficiency

    """
    # Efficiency depends on time
    t_eff = _time_efficiency(dataframe["time"], time_params)

    # Also pT
    pts = tuple(
        _pt(dataframe[f"{p}_PX"], dataframe[f"{p}_PY"])
        for p in (
            "Dst_ReFit_D0_Kplus",
            "Dst_ReFit_D0_piplus",
            "Dst_ReFit_D0_piplus_0",
            "Dst_ReFit_D0_piplus_1",
        )
    )
    pt_eff = np.multiply.reduce(
        [
            eff(pt)
            for (eff, pt) in zip(
                (
                    _k_efficiency,
                    _piminus_efficiency,
                    _piminus_efficiency,
                    _piplus_efficiency,
                ),
                pts,
            )
        ]
    )

    eff = pt_eff * t_eff
    return np.random.random(len(eff)) < eff


def _mass_pts(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Phsp + time pts with inv masses

    """
    k, pi1, pi2, pi3 = util.k_3pi(dataframe)
    pi1, pi2 = util.momentum_order(k, pi1, pi2)
    return np.column_stack((inv_mass_param(k, pi1, pi2, pi3), dataframe["time"]))


def _pts(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Phase space + time points

    """
    k, pi1, pi2, pi3 = util.k_3pi(dataframe)
    pi1, pi2 = util.momentum_order(k, pi1, pi2)
    return np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))


def _plot_points(
    target_pts: np.ndarray, orig_pts: np.ndarray, weights: np.ndarray
) -> Tuple[plt.Figure, dict]:
    """
    returns figure, dict of axes

    """
    fig, ax = plt.subplot_mosaic(
        """
        AAABBBCCC
        AAABBBCCC
        AAABBBCCC
        DDDEEEFFF
        .........
        GGGHHHIII
        GGGHHHIII
        GGGHHHIII
        JJJKKKLLL
        """,
        figsize=(15, 10),
        gridspec_kw={"hspace": 0.0},
    )
    hist_kw = {"density": True, "histtype": "step"}

    for axis, pull_axis, target, orig in zip(
        (ax[l] for l in "ABCGHI"), (ax[l] for l in "DEFJKL"), target_pts.T, orig_pts.T
    ):
        # Find bin limits
        # Horrible, hacky
        bin_min = min(np.min(target), np.min(orig))
        if bin_min < 0:
            bin_min *= 1.0001
        else:
            bin_min *= 0.9999
        bin_max = max(np.max(target), np.max(orig))
        if bin_max < 0:
            bin_max *= 0.9999
        else:
            bin_max *= 1.0001
        bins = np.linspace(
            bin_min,
            bin_max,
            100,
        )
        axis.hist(orig, bins=bins, label="Original", **hist_kw, alpha=0.5)
        contents, _, _ = axis.hist(target, bins=bins, label="Target", **hist_kw)
        axis.hist(orig, bins=bins, label="Reweighted", **hist_kw, weights=weights)
        axis.set_ylim(0, np.max(contents) * 1.1)

        pull(pull_axis, bins, (target, orig), (None, None))
        pull(pull_axis, bins, (target, orig), (None, weights))
        pull_axis.plot(pull_axis.get_xlim(), (0, 0), "k-")

    ax["A"].legend()
    # Big axes only
    for axis in (ax[l] for l in "ABCGHI"):
        axis.set_xticks([])
        axis.set_ylabel("Counts")

    # Small axes only
    for axis in (ax[l] for l in "DEFJK"):
        axis.set_ylim(-0.0023, 0.0023)

    for axis in ax.values():
        axis.set_yticks([])

    return fig, ax


def _reweight(
    dataframe: pd.DataFrame, sign: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the efficiency, reweight, make some plots +
    return the testing (target, original, weights)

    """
    target_df = dataframe[~dataframe["train"]]
    orig_df = target_df[target_df["accepted"]]

    # Get the reweighter from pickle dump
    reweighter = eff_get.ampgen_reweighter_dump(sign, verbose=True)

    # Get phase space points
    target_pts = _pts(target_df)
    orig_pts = _pts(orig_df)

    # Plot projections
    weights = reweighter.weights(orig_pts)
    fig, ax = _plot_points(target_pts, orig_pts, weights)
    for axis, label in zip((ax[l] for l in "DEFJKL"), plotting.phsp_labels()):
        axis.set_xlabel(label)

    ax["L"].set_xlabel(r"$t/\tau$")

    fig.tight_layout()
    path = f"toy_test_proj_{sign}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    target_masses = _mass_pts(target_df)
    orig_masses = _mass_pts(orig_df)
    fig, ax = _plot_points(target_masses, orig_masses, weights)
    for axis, label in zip(
        (ax[l] for l in "DEFJKL"),
        (
            r"$M(K^+\pi_1^-)/MeV$",
            r"$M(\pi_1^-\pi_2^-)/MeV$",
            r"$M(\pi_2^-\pi_3^+)/MeV$",
            r"$M(K^+\pi_1^-\pi_2^-)/MeV$",
            r"$M(\pi_1^-\pi_2^-\pi_3^+)/MeV$",
            r"$t/\tau$",
        ),
    ):
        axis.set_xlabel(label)

    fig.tight_layout()
    path = f"toy_test_masses_{sign}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    # Plot Z scatter
    fig, ax, _, _ = plotting.z_scatter(
        *util.k_3pi(target_df),
        *util.k_3pi(orig_df),
        weights,
        np.ones(len(weights)),
        5,
    )

    fig.tight_layout()
    path = f"toy_test_z_{sign}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    return target_pts, orig_pts, weights


def main():
    """
    Read AmpGen dataframes, apply an efficiency, measure how good it was

    """
    # Read dataframes
    print("getting dfs")
    cf_df = get.ampgen("cf")
    dcs_df = get.ampgen("dcs")

    # Only keep stuff above min time
    print("cutting dfs (time)")
    cf_df = cf_df[cf_df["time"] > efficiency_definitions.MIN_TIME]
    dcs_df = dcs_df[dcs_df["time"] > efficiency_definitions.MIN_TIME]

    cf_target, cf_orig, cf_wt = _reweight(cf_df, "cf")
    dcs_target, dcs_orig, dcs_wt = _reweight(dcs_df, "dcs")

    # Plot ratio
    print("plotting ratio")
    fig, ax = plotting.plot_ratios(
        cf_orig[:, -1],
        dcs_orig[:, -1],
        cf_target[:, -1],
        dcs_target[:, -1],
        cf_wt,
        dcs_wt,
        TIME_BINS[2:],
        np.ones(len(cf_orig)),
        np.ones(len(dcs_orig)),
    )
    ax[0].set_title("Original")
    ax[1].set_title("Target")
    ax[2].set_title("Reweighted")

    for a in ax[1:]:
        ax[1].set_yticks([])
        ax[2].set_yticks([])

    for a in ax:
        for text in a.texts:
            text.remove()
    fig.savefig("toy_test_ratio.png")


if __name__ == "__main__":
    main()
