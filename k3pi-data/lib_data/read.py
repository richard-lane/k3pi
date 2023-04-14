"""
Utilities for reading data and stuff

"""
import pathlib
from typing import List

import pandas as pd


def _branch_file() -> pathlib.Path:
    """
    Where the branch names live

    """
    return pathlib.Path(__file__).resolve().parents[1] / "branch_names.txt"


def _all_branches() -> List[str]:
    """All branch names of interest"""
    with open(str(_branch_file()), "r", encoding="utf8") as txt_f:
        return [line.strip() for line in txt_f.readlines()]


def branches(data_type: str) -> List[str]:
    """
    Get the branches we need to skim from a ROOT file

    :param data_type: "pgun", "MC", or "data" - tells us which branches to use
                      (e.g. theres no BKG cat info for data)

    """
    assert data_type in {"data", "pgun", "MC"}

    lines = _all_branches()

    if data_type == "data":
        return lines[:41] + lines[43:49]

    elif data_type == "pgun":
        # No tracking or HLT information in the pgun data files
        return lines[:43] + lines[-1:]

    # MC
    return lines[:49]


def pgun_hlt_branches() -> List[str]:
    """
    Get the particle gun HLT branch names

    """
    return _all_branches()[-3:-1]


def remove_refit(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Take only the first (best) result from the ReFit

    """
    return dataframe.groupby(level=[0]).first()
