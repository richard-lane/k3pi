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


def branches(data_type: str) -> List[str]:
    """
    Get the branches we need to skim from a ROOT file

    :param data_type: "pgun", "MC", or "data" - tells us which branches to use
                      (e.g. theres no BKG cat info for data)

    """
    assert data_type in {"data", "pgun", "MC"}

    with open(str(_branch_file()), "r", encoding="utf8") as txt_f:
        lines = [line.strip() for line in txt_f.readlines()]

    if data_type == "data":
        return lines[:41]

    elif data_type == "pgun":
        return lines[:39] + lines[-4:]

    return lines[:42]


def remove_refit(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Take only the first (best) result from the ReFit

    """
    return dataframe.groupby(level=[0]).first()
