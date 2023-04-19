"""
Add phsp bin index column to existing dataframes

"""
import sys
import glob
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, phsp_binning


def _bin_numbers(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Find the phase space bin number for each event in a dataframe

    """
    # Find the amplitude cross term for each event
    interference_terms = phsp_binning.interference_terms(dataframe)

    # Find the corresponding angle
    angles = np.angle(interference_terms, deg=True)

    # Bin these angles in terms of phsp bins
    indices = np.digitize(angles, phsp_binning.BINS) - 1

    # use the pipi masses to veto around the Ks mass
    veto = phsp_binning.ks_veto(dataframe)

    # Set veto'd events bin index to a special value
    indices[veto] = 4

    return indices


def main(*, year: str, sign: str, magnetisation: str) -> None:
    """
    Add phsp bin info to dataframes

    """
    dump_paths = glob.glob(
        str(definitions.data_dir(year, sign, magnetisation) / "*pkl")
    )

    col_header = "phsp bin"
    for path in tqdm(dump_paths):
        # Open the dataframe
        with open(path, "rb") as f:
            dataframe = pickle.load(f)

        if col_header in dataframe:
            continue

        # Find which events are K+ type, and the K+ and K- type interference terms
        bin_indices = _bin_numbers(dataframe)

        # Add this column to the dataframe
        dataframe[col_header] = bin_indices

        # Dump it
        with open(path, "wb") as f:
            pickle.dump(dataframe, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a column for phase space bin index to the DataFrames to `dumps/`"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
