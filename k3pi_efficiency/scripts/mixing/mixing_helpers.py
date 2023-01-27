"""
Helper fcns for the mixing scripts, because lots of code is being
repeated

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from lib_efficiency import mixing, efficiency_util
from lib_efficiency.amplitude_models import amplitudes


def mixing_weights(
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    r_d: float,
    params: mixing.MixingParams,
    k_sign: int,
    q_p: Tuple[float, float],
):
    """
    Find weights to apply to the dataframe to introduce mixing

    """
    assert k_sign in {"both", "k_plus", "k_minus"}

    dcs_lifetimes = dcs_df["time"]

    # Need to find the right amount to scale the amplitudes by
    dcs_scale = r_d / np.sqrt(amplitudes.DCS_AVG_SQ)
    cf_scale = 1 / np.sqrt(amplitudes.CF_AVG_SQ)
    denom_scale = 1 / np.sqrt(amplitudes.DCS_AVG_SQ)

    # Simple case where we only have K+ or K- in our dataframe(s)
    if not k_sign == "both":
        dcs_k3pi = efficiency_util.k_3pi(dcs_df)
        weights = mixing.ws_mixing_weights(
            dcs_k3pi,
            dcs_lifetimes,
            params,
            +1 if k_sign == "k_plus" else -1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

    # More complicated case where we could have both
    else:
        weights = np.ones(len(dcs_df)) * np.inf
        plus_mask = dcs_df["K ID"] == 321

        # K+ weights
        print(f"finding {np.sum(plus_mask)} k+ mixing weights")
        weights[plus_mask] = mixing.ws_mixing_weights(
            efficiency_util.k_3pi(dcs_df[plus_mask]),
            dcs_lifetimes[plus_mask],
            params,
            +1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

        # K- weights
        print(f"finding {len(plus_mask) - np.sum(plus_mask)} k- mixing weights")
        weights[~plus_mask] = mixing.ws_mixing_weights(
            efficiency_util.k_3pi(dcs_df[~plus_mask]),
            dcs_lifetimes[~plus_mask],
            params,
            -1,
            q_p,
            dcs_scale=dcs_scale,
            cf_scale=cf_scale,
            denom_scale=denom_scale,
        )

    # Scale weights such that their mean is right
    scale = len(dcs_df) / len(cf_df)

    return weights * scale
