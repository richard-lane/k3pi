"""
Definitions of stuff needed for creating/using the efficiency reweighter

"""
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from lib_time_fit import definitions


REWEIGHTER_DIR = pathlib.Path(__file__).resolve().parents[1] / "reweighter"

# Time (in lifetimes) below which we just throw away events - the reweighting here is too
# unstable
MIN_TIME = definitions.TIME_BINS[2]

# Absolute efficiencies from particle gun
# Deprecated;
RS_EFF = 2.291 * np.nan
RS_ERR = 0.003 * np.nan
WS_EFF = 2.247 * np.nan
WS_ERR = 0.003 * np.nan
FALSE_EFF = 2.267 * np.nan
FALSE_ERR = 0.007 * np.nan


def reweighter_path(
    year: str,
    sign: str,
    magnetisation: str,
    k_sign: str,
    time_fit: bool,
    cut: bool,
) -> pathlib.Path:
    """
    Where the efficiency correction reweighter lives

    """
    assert year in {"2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magdown"}
    assert k_sign in {"k_plus", "k_minus", "both"}

    suffix = "_time_fit" if time_fit else ""
    suffix = f"{'_time_fit' if time_fit else ''}{'bdt_cut' if cut else ''}"

    return REWEIGHTER_DIR / f"{year}_{sign}_{magnetisation}_{k_sign}{suffix}.pkl"


def ampgen_reweighter_path(sign: str) -> pathlib.Path:
    """
    Where the efficiency correction reweighter lives

    """
    assert sign in {"cf", "dcs"}
    return REWEIGHTER_DIR / f"ampgen_{sign}.pkl"


def reweighter_exists(
    year: str, sign: str, magnetisation: str, k_sign: str, time_fit: bool, cut: bool
) -> bool:
    """
    Whether the reweighter has been created yet

    """
    return reweighter_path(year, sign, magnetisation, k_sign, time_fit, cut).is_file()
