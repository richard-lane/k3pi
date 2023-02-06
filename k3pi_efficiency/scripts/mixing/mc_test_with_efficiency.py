"""
Introduce some mixing to the MC dataframes via weighting

Also option to do the efficiency correction to see what difference it makes

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
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_signal_cuts"))
import pdg_params
import mixing_helpers
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_definitions import MIN_TIME
from lib_time_fit import util as fit_util
from lib_data import get, definitions, stats
from lib_data import util as data_util
from lib_cuts import get as cuts_get
from lib_cuts.definitions import THRESHOLD


def _bdt_cut_df(
    dataframe: pd.DataFrame, year: str, sign: str, magnetisation: str
) -> pd.DataFrame:
    """
    Perform BDT cut on a dataframe

    """
    clf = cuts_get.classifier(year, sign, magnetisation)
    return cuts_get.signal_cut_df(dataframe, clf, THRESHOLD)


def _efficiency_weights(
    dataframe: pd.DataFrame, year: str, magnetisation: str, sign: str, k_sign: str
) -> np.ndarray:
    """
    Get efficiency weights for a dataframe

    """
    # Get the right reweighter
    reweighter = get_reweighter(
        year, sign, magnetisation, k_sign="both", fit=False, cut=False
    )

    # Get the k3pi
    k, pi1, pi2, pi3 = efficiency_util.k_3pi(dataframe)
    # Momentum order the pions
    pi1, pi2 = data_util.momentum_order(k, pi1, pi2)

    return reweighter.weights(
        efficiency_util.points(k, pi1, pi2, pi3, dataframe["time"])
    )


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
    year: str,
    magnetisation: str,
    k_sign: str,
    eff: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error, with efficiency correction

    """
    # Get efficiency weights from the dataframes
    if eff:
        cf_eff_wt = _efficiency_weights(cf_df, year, magnetisation, "cf", k_sign)
        dcs_eff_wt = _efficiency_weights(dcs_df, year, magnetisation, "dcs", k_sign)

        # Scale efficiency weights
        cf_eff_wt, dcs_eff_wt = (
            arr[0]
            for arr in efficiency_util.scale_weights(
                [cf_eff_wt], [dcs_eff_wt], 0.98  # from particle gun
            )
        )
    else:
        cf_eff_wt = np.ones(len(cf_df))
        dcs_eff_wt = np.ones(len(dcs_df))

    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins, weights=cf_eff_wt)
    dcs_counts, dcs_errs = stats.counts(
        dcs_df["time"], bins=bins, weights=dcs_wt * dcs_eff_wt
    )

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


def _time_cut(dataframe: pd.DataFrame, max_time: float) -> pd.DataFrame:
    """
    Do the time cut on a dataframe

    """
    keep = (MIN_TIME < dataframe["time"]) & (dataframe["time"] < max_time)

    return dataframe.loc[keep]


def main(args: argparse.Namespace):
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times with efficiency correction

    """
    k_sign = args.k_sign

    # Read MC dataframes
    year, magnetisation = "2018", "magdown"
    cf_df = get.mc(year, "cf", magnetisation)
    dcs_df = get.mc(year, "dcs", magnetisation)

    # Time cuts
    max_time = 7
    cf_df = _time_cut(cf_df, max_time)
    dcs_df = _time_cut(dcs_df, max_time)

    # Select K signs
    if not k_sign == "both":
        assert np.all(np.abs(dcs_df["K ID"]) == 321), "Check DCS K ids all +-321"
        assert np.all(np.abs(cf_df["K ID"]) == 321), "Check CF K ids all +-321"

        if k_sign == "k_plus":
            cf_keep = cf_df["K ID"] > 0
            dcs_keep = dcs_df["K ID"] > 0
        else:
            cf_keep = cf_df["K ID"] < 0
            dcs_keep = dcs_df["K ID"] < 0

        cf_df = cf_df.loc[cf_keep]
        dcs_df = dcs_df.loc[dcs_keep]

    if not k_sign == "both":
        print(f"{k_sign=}\t{len(dcs_df)} evts")

    bins = np.linspace(0, max_time, 15)

    # Convert dtype of kinematic columns
    cf_df = cf_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})
    dcs_df = dcs_df.astype({k: np.float64 for k in definitions.MOMENTUM_COLUMNS})

    # Perform BDT cuts if needed
    if args.bdt:
        print("performing bdt cut")
        cf_df = _bdt_cut_df(cf_df, year, "dcs", magnetisation)
        dcs_df = _bdt_cut_df(dcs_df, year, "dcs", magnetisation)

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

    # Find mixed ratio and error
    ratio, err = _ratio_err(
        bins,
        cf_df,
        dcs_df,
        mixing_weights,
        year,
        magnetisation,
        k_sign,
        args.eff,
    )

    # Cut off the first bin because it's bad
    ratio = ratio[1:]
    err = err[1:]
    bins = bins[1:]

    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=amplitudes.AMPGEN_Z.real,
        im_z=amplitudes.AMPGEN_Z.imag,
    )

    # Make a plot showing fits and scan
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    path = f"mc_mixed_fits_{k_sign}{'_eff' if args.eff else ''}{'_bdt' if args.bdt else ''}.png"
    mixing_helpers.scan_fits(fig, axes, ratio, err, bins, ideal, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduce D mixing via weighting to the LHCb MC dataframes"
    )

    parser.add_argument(
        "k_sign",
        help="whether to use K+ or K- type events (or both)",
        choices={"k_plus", "k_minus", "both"},
    )
    parser.add_argument(
        "--eff",
        help="Whether to do an efficiency correction.",
        action="store_true",
    )
    parser.add_argument(
        "--bdt",
        help="Whether to do the BDT cut.",
        action="store_true",
    )

    main(parser.parse_args())
