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
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi_fitter"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
import pdg_params
import mixing_helpers
from lib_efficiency import efficiency_util, mixing
from lib_efficiency.amplitude_models import amplitudes
from lib_time_fit import util as fit_util
from lib_time_fit import fitter, plotting
from lib_data import stats


def _ratio_err(
    bins: np.ndarray,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio and error

    """
    cf_counts, cf_errs = stats.counts(cf_df["time"], bins=bins)
    dcs_counts, dcs_errs = stats.counts(dcs_df["time"], bins=bins, weights=dcs_wt)

    return fit_util.ratio_err(dcs_counts, cf_counts, dcs_errs, cf_errs)


def main(args):
    """
    Read MC dataframes, add some mixing to the DCS frame, plot the ratio
    of the decay times

    """
    # Read AmpGen dataframes
    k_sign = args.k_sign

    cf_df = efficiency_util.ampgen_df("cf", k_sign, train=None)
    dcs_df = efficiency_util.ampgen_df("dcs", k_sign, train=None)

    # Time cut
    max_time = 15
    cf_keep = (0 < cf_df["time"]) & (cf_df["time"] < max_time)
    dcs_keep = (0 < dcs_df["time"]) & (dcs_df["time"] < max_time)
    cf_df = cf_df[cf_keep]
    dcs_df = dcs_df[dcs_keep]

    print(f"{len(cf_df)=}\t{len(dcs_df)=}")

    bins = np.linspace(0, max_time, 20)

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
    ratio, err = _ratio_err(bins, cf_df, dcs_df, mixing_weights)

    ideal = fit_util.ScanParams(
        r_d=r_d,
        x=params.mixing_x,
        y=params.mixing_y,
        re_z=amplitudes.AMPGEN_Z.real,
        im_z=amplitudes.AMPGEN_Z.imag,
    )

    # Make a plot showing fits and scan
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    path = f"ampgen_mixed_fits_{k_sign}.png"
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
    main(parser.parse_args())
