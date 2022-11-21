"""
Utilitility fractions

"""
import sys
import pathlib
from typing import Tuple, Iterable
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import stats


def delta_m(dataframe: pd.DataFrame) -> pd.Series:
    """
    Get mass difference from a dataframe

    """
    return dataframe["D* mass"] - dataframe["D0 mass"]


def delta_m_generator(dataframes: Iterable[pd.DataFrame]) -> Iterable[pd.Series]:
    """
    Get generator of mass differences from an iterable of dataframes

    """
    for dataframe in dataframes:
        yield delta_m(dataframe)


def delta_m_counts(
    dataframes: Iterable[pd.DataFrame],
    bins: np.ndarray,
    weights: Iterable[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return binned delta M from an iterable of dataframes

    """
    return stats.counts_generator(delta_m_generator(dataframes), bins, weights)
