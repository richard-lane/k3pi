"""
Functions to perform cuts on a dataframe; all have the same signature

"""
import pandas as pd


def delta_m(df: pd.DataFrame) -> pd.DataFrame:
    d_m = df["Delta_M"]
    return df[(139.0 < d_m) & (d_m < 152.0)].dropna()


def d_mass(df: pd.DataFrame) -> pd.DataFrame:
    m = df["D0_M_ReFit"]
    return df[(1840.0 < m) & (m < 1888.0)].dropna()


def p_t(df: pd.DataFrame) -> pd.DataFrame:
    # Daughters must have pT > 250
    daughters = "K_PT", "pi0_PT", "pi1_PT", "pi_PT"

    retval = df
    for daughter in daughters:
        retval = retval[retval[daughter] > 250].dropna()

    # Slow pi must have pT > 200
    slow_pi = "Dst_ReFit_piplus_PT"
    return retval[retval[slow_pi] > 200].dropna()


def all_cuts(df: pd.DataFrame) -> pd.DataFrame:
    return delta_m(d_mass(p_t(df)))
