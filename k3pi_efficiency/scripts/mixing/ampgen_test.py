"""
Introduce some mixing to the AmpGen dataframes via weighting

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
import pdg_params
import mixing_helpers
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency.get import ampgen_reweighter_dump as get_reweighter
from lib_time_fit import util as fit_util
from lib_time_fit.definitions import TIME_BINS
from lib_data import stats, util as data_util


def _efficiency_weights(
    dataframe: pd.DataFrame, sign: str, *, correct_efficiency: bool
) -> np.ndarray:
    """
    Get efficiency weights for a dataframe

    """
    if not correct_efficiency:
        return np.ones(len(dataframe))

    # Get the right reweighter
    reweighter = get_reweighter(sign, verbose=True)

    # Get the k3pi
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)

    # Momentum order the pions
    pi1, pi2 = data_util.momentum_order(k, pi1, pi2)

    weights = reweighter.weights(
        efficiency_util.points(k, pi1, pi2, pi3, dataframe["time"])
    )
    return weights


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
    abs_eff_ratio: float,
    *,
    correct_efficiency: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error

    """
    if not correct_efficiency:
        # Don't need if if we're not correcting
        assert abs_eff_ratio is None

    # Get efficiency weights if we need
    cf_eff_wt = _efficiency_weights(cf_df, "cf", correct_efficiency=correct_efficiency)
    dcs_eff_wt = _efficiency_weights(
        dcs_df, "dcs", correct_efficiency=correct_efficiency
    )

    # Scale
    # if correct_efficiency:
    #     cf_eff_wt, dcs_eff_wt = (
    #         arr[0]
    #         for arr in efficiency_util.scale_weights(
    #             [cf_eff_wt], [dcs_eff_wt], abs_eff_ratio
    #         )
    #     )

    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins, weights=cf_eff_wt)
    dcs_counts, dcs_errs = stats.counts(
        dcs_df["time"], bins=bins, weights=dcs_wt * dcs_eff_wt
    )

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


def _abs_eff(dataframe: pd.DataFrame, wts: np.ndarray = None) -> float:
    """
    Absolute efficiency

    """
    accepted = dataframe["accepted"]
    if wts is None:
        accepted = np.sum(accepted)
        total = len(dataframe)

    else:
        accepted = np.sum(wts[accepted])
        total = np.sum(wts)

    efficiency = accepted / total
    print(f"{accepted=}\t{total=}\t{efficiency=}")
    return efficiency


def main(args):
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times

    """
    k_sign = args.k_sign

    if args.correct_efficiency:
        assert args.apply_efficiency

    # Read AmpGen dataframes
    cf_df = efficiency_util.ampgen_df("cf", k_sign, train=None)
    dcs_df = efficiency_util.ampgen_df("dcs", k_sign, train=None)

    # Potentially throw some stuff away
    if args.scale is not None:
        print(f"cf scale {args.scale}")
        cf_df = cf_df[: int(len(cf_df) * args.scale)]

    # Time cut
    max_time = TIME_BINS[-2]
    cf_keep = (MIN_TIME < cf_df["time"]) & (cf_df["time"] < max_time)
    dcs_keep = (MIN_TIME < dcs_df["time"]) & (dcs_df["time"] < max_time)
    cf_df = cf_df[cf_keep]
    dcs_df = dcs_df[dcs_keep]

    # Parameters determining mixing
    r_d = np.sqrt(0.003025)
    print(f"{r_d**2=}")
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=pdg_params.mixing_x(),
        mixing_y=pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = mixing_helpers.mixing_weights(
        cf_df, dcs_df, r_d, params, k_sign, q_p
    )

    # Find the absolute efficiencies from the dataframes after
    # the time cuts
    if args.correct_efficiency:
        dcs_abs_eff = _abs_eff(dcs_df, mixing_weights)
        cf_abs_eff = _abs_eff(cf_df)

    # Apply the efficiency via the boolean mask
    if args.apply_efficiency:
        cf_df = cf_df[cf_df["accepted"]]

        mixing_weights = mixing_weights[dcs_df["accepted"]]
        dcs_df = dcs_df[dcs_df["accepted"]]

    print(f"{len(cf_df)=}\t{len(dcs_df)=}")

    # Find mixed ratio and error
    bins = TIME_BINS[2:-1]
    ratio, err = _ratio_err(
        bins,
        cf_df,
        dcs_df,
        mixing_weights,
        dcs_abs_eff / cf_abs_eff if args.correct_efficiency else None,
        correct_efficiency=args.correct_efficiency,
    )

    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=amplitudes.AMPGEN_Z.real,
        im_z=amplitudes.AMPGEN_Z.imag,
    )

    # Make a plot showing fits and scan
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    path = (
        f"ampgen_mixed_fits_{k_sign}"
        f"{'_eff' if args.apply_efficiency else ''}"
        f"{'_corrected' if args.correct_efficiency else ''}.png"
    )
    mixing_helpers.scan_fits(fig, axes, ratio, err, bins, ideal, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduce D mixing via weighting to the AmpGen dataframes"
    )

    parser.add_argument(
        "k_sign",
        help="whether to use K+ or K- type events (or both)",
        choices={"k_plus", "k_minus", "both"},
    )

    parser.add_argument(
        "--apply_efficiency",
        help="whether to apply a mock efficiency model",
        action="store_true",
    )

    parser.add_argument(
        "--correct_efficiency",
        help="whether to correct for the efficiency",
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--scale",
        help="tells us how much of the cf df to throw away",
        type=float,
    )

    main(parser.parse_args())
