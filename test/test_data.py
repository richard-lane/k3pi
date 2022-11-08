"""
k3pi-data unit test

"""
import sys
import pytest
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))

from lib_data import definitions
from lib_data.util import flip_momenta


def test_flip_momenta():
    """
    Check we flip momenta correctly

    """
    # Make a toy dataframe
    n_labels = len(definitions.MOMENTUM_COLUMNS)
    dataframe = pd.DataFrame(
        dict(zip(definitions.MOMENTUM_COLUMNS, [[l, l * 2] for l in range(n_labels)]))
    )

    # expected df after flipping k3pi 3 momenta from the first row
    to_flip = np.array([True, False], dtype=np.bool_)
    expected_df = dataframe.copy()

    # change sign for k3pi 3 momenta in the first row
    k3pi_3momenta = expected_df.columns.str.contains(
        "us"
    ) & ~expected_df.columns.str.endswith("_E")
    expected_df.loc[0, k3pi_3momenta] *= -1

    # check that they're the same
    dataframe = flip_momenta(dataframe, to_flip=to_flip)
    assert dataframe.iloc[0].equals(expected_df.iloc[0])
    assert dataframe.iloc[1].equals(expected_df.iloc[1])
