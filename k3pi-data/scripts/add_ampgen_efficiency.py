"""
Add a column to the AmpGen dataframes for whether each event
was detected or not under some mock efficiency function

"""
import sys
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_efficiency import mock_efficiency
from lib_data import get, definitions


def _efficiency_column_exists(dataframe: pd.DataFrame) -> bool:
    """whether the efficiency column is already there"""
    return "accepted" in dataframe


def main(sign: str):
    """
    Use rejection sampling to determine whether each event was accepted
    under some mock efficiency function

    Then add columns to the dataframe for whether each event was accepted/rejected

    Finally write the dataframes to disk

    """
    rng = np.random.default_rng(seed=0)

    # Read AmpGen dataframe
    dataframe = get.ampgen(sign)

    if _efficiency_column_exists(dataframe):
        print(f"overwriting {sign} column")

    # Use rejection sampling to see which events are kept
    time_factor = 0.98 if sign == "dcs" else 1.0
    phsp_factor = 1.0 if sign == "dcs" else 0.98
    kept = mock_efficiency.accepted(rng, dataframe, time_factor=time_factor, phsp_factor=phsp_factor)
    print(f"{np.sum(kept)=}\t{len(kept)}")

    # Add a column to the dataframe showing this
    dataframe["accepted"] = kept

    # Write to disk
    with open(definitions.ampgen_dump(sign), "wb") as f:
        pickle.dump(dataframe, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add efficiency and test/train columns to AmpGen dataframe. Also plot stuff"
    )
    parser.add_argument("sign", choices={"dcs", "cf"}, help="Which dataframe to use")

    main(parser.parse_args().sign)
