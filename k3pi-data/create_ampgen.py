"""
Create AmpGen dataframes

The AmpGen files are small enough that we can store the entire thing in one DataFrame (probably -
it depends how many events you generate; I tested it with 1,000,000)

Should use D -> K+3pi AmpGen models

"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import uproot

from lib_data import definitions
from lib_data import util


def _ampgen_df(gen: np.random.Generator, tree, sign: str) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta and time arrays from the provided tree

    """
    df = pd.DataFrame()

    t_branch = "Dbar0_decayTime" if sign == "cf" else "D0_decayTime"

    # There are no ReFit variables or any entries in the tree with multiple candidates
    # so we can just use numpy to read everything in nicely
    # Simply passing library="np" here seems to do the right thing, since apparently
    # setting uproot.default_library = "np" didn't work...
    # TODO: consistent AmpGen time
    df["time"] = (
        tree[t_branch].array(library="np") * 1000 / 0.41
    )  # Convert to d lifetimes
    # TODO should use a constant somewhere to encode
    # the actual AmpGen lifetime

    # Expect to have K+3pi AmpGen
    # This is why we have K~ (K+) and two pi# (pi-)
    branches = [
        *(f"_1_K~_{s}" for s in definitions.MOMENTUM_SUFFICES),
        *(f"_2_pi#_{s}" for s in definitions.MOMENTUM_SUFFICES),
        *(f"_3_pi#_{s}" for s in definitions.MOMENTUM_SUFFICES),
        *(f"_4_pi~_{s}" for s in definitions.MOMENTUM_SUFFICES),
    ]

    for branch, column in zip(branches, definitions.MOMENTUM_COLUMNS):
        df[column] = tree[branch].array() * 1000  # Convert to MeV

    util.add_train_column(gen, df)

    return df


def main(path: str, sign: str) -> None:
    """Create a DataFrame holding AmpGen momenta"""
    # If the dir doesnt exist, create it
    if not definitions.AMPGEN_DIR.is_dir():
        os.mkdir(definitions.AMPGEN_DIR)

    # We need a random number generator to decide which elements are test/train
    gen = np.random.default_rng()

    # Read the dataframe
    with uproot.open(path) as ag_f:
        dataframe = _ampgen_df(gen, ag_f["DalitzEventList"], sign)

    # Dump it
    with open(definitions.ampgen_dump(sign), "wb") as f:
        pickle.dump(dataframe, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "path", type=str, help="path to AmpGen-generated D->K+3pi ROOT file"
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF.",
    )

    args = parser.parse_args()

    main(args.path, args.sign)
