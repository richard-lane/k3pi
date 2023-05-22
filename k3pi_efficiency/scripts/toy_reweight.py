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
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)
    pi1, pi2 = util.momentum_order(k, pi1, pi2)
    return np.column_stack((inv_mass_param(k, pi1, pi2, pi3), dataframe["time"]))


def _pts(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Phase space + time points

    """
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)
    pi1, pi2 = util.momentum_order(k, pi1, pi2)
    return np.column_stack((helicity_param(k, pi1, pi2, pi3), dataframe["time"]))


def _plot_efficiencies(time_params):
    """Line plots of analytic efficiency"""
    t_pts = np.linspace(0, 10, 100)
    pt_pts = np.linspace(0, 800, 100)

    t_eff = _time_efficiency(t_pts, time_params)

    k_eff = _k_efficiency(pt_pts)
    pi1_eff = _piminus_efficiency(pt_pts)
    pi2_eff = _piminus_efficiency(pt_pts)
    pi3_eff = _piplus_efficiency(pt_pts)
    p_eff = np.multiply.reduce((k_eff, pi1_eff, pi2_eff, pi3_eff))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    kw = {"linestyle": "--", "alpha": 0.8}

    ax[0].plot(t_pts, t_eff, "k-", label=r"$\epsilon(t)$")

    ax[1].plot(pt_pts, k_eff, **kw, label=r"$\epsilon(K)$")
    ax[1].plot(pt_pts, pi1_eff, **kw, linewidth=2.5, label=r"$\epsilon(\pi_1^-)$")
    ax[1].plot(pt_pts, pi2_eff, **kw, label=r"$\epsilon(\pi_2^-)$")
    ax[1].plot(pt_pts, pi3_eff, **kw, label=r"$\epsilon(\pi^+)$")
    ax[1].plot(pt_pts, p_eff, "k-", label=r"$\epsilon(p)$")

    ax[0].set_ylim(0, 1)

    ax[0].legend()
    ax[1].legend()

    ax[0].set_xlabel(r"$t / \tau$")
    ax[1].set_xlabel(r"$p_T$ /MeV")
    ax[0].set_ylabel(r"Efficiency")

    fig.tight_layout()
    fig.savefig("toy_efficiency.png")


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
    for axis in (ax[l] for l in "ABCGHI"):
        axis.set_xticks([])
    for axis in (ax[l] for l in "DEFJK"):
        axis.set_ylim(-0.0023, 0.0023)

    return fig, ax


def _reweight(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the efficiency, reweight, make some plots +
    return the testing (target, original, weights)

    """
    train = dataframe[dataframe["train"]]
    test = dataframe[~dataframe["train"]]

    # Plot efficiencies
    rng = np.random.default_rng()
    time_params = (
        rng.normal(loc=1, scale=0.1),
        rng.normal(loc=2, scale=0.2),
        rng.normal(loc=1, scale=0.1),
        rng.normal(loc=2, scale=0.2),
    )
    _plot_efficiencies(time_params)

    # Apply efficiencies
    keep = _efficiency(train, time_params)

    # Perform reweighting
    target_pts = _pts(train)
    orig_pts = target_pts[keep]
    train_kwargs = {
        "n_estimators": 10,
        "max_depth": 5,
        "learning_rate": 0.7,
        "min_samples_leaf": 1800,
        "n_bins": 10000,
    }
    reweighter = EfficiencyWeighter(
        target_pts,
        orig_pts,
        original_weight=np.ones(len(orig_pts)),
        fit=False,
        min_t=efficiency_definitions.MIN_TIME,
        **train_kwargs,
    )

    # Apply the efficiency to the test set
    test_keep = _efficiency(test, time_params)
    target_pts = _pts(test)
    orig_pts = target_pts[test_keep]

    # Plot projections
    weights = reweighter.weights(orig_pts)
    fig, ax = _plot_points(target_pts, orig_pts, weights)
    for axis, label in zip((ax[l] for l in "DEFJKL"), plotting.phsp_labels()):
        axis.set_xlabel(label)
    ax["L"].set_xlabel(r"$t/\tau$")
    fig.tight_layout()
    fig.savefig("toy_test_proj.png")

    target_masses = _mass_pts(test)
    orig_masses = target_masses[test_keep]
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
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig("toy_test_masses.png")

    # Plot Z scatter
    fig, ax, _, _ = plotting.z_scatter(
        *efficiency_util.k_3pi(test),
        *efficiency_util.k_3pi(test[test_keep]),
        weights,
        np.ones(np.sum(test_keep)),
        5,
    )
    fig.savefig("toy_test_z.png")

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

    cf_target, cf_orig, cf_wt = _reweight(cf_df)
    dcs_target, dcs_orig, dcs_wt = _reweight(cf_df)

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
    for a in ax:
        for text in a.texts:
            text.remove()
    fig.savefig("toy_test_ratio.png")


if __name__ == "__main__":
    main()
