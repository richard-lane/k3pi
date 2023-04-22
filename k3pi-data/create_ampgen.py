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

    df["time"] = (
        tree[t_branch].array(library="np") * 1000 / definitions.D0_LIFETIME_PS
    )  # Convert to d lifetimes

    # Expect to have K+3pi AmpGen
    # This is why we have K~ (K+) and two pi# (pi-)
    momentum_suffices = "Px", "Py", "Pz", "E"
    branches = [
        *(f"_1_K~_{s}" for s in momentum_suffices),
        *(f"_2_pi#_{s}" for s in momentum_suffices),
        *(f"_3_pi#_{s}" for s in momentum_suffices),
        *(f"_4_pi~_{s}" for s in momentum_suffices),
    ]

    for branch, column in zip(branches, definitions.MOMENTUM_COLUMNS):
        df[column] = tree[branch].array(library="np") * 1000  # Convert to MeV

    util.add_train_column(gen, df)

    return df


def main(path: str, sign: str) -> None:
    """Create a DataFrame holding AmpGen momenta"""
    # If the dump exists, don't do anything
    dump_path = definitions.ampgen_dump(sign)
    if dump_path.is_file():
        return

    # If the dir doesnt exist, create it
    if not definitions.AMPGEN_DIR.is_dir():
        # This can occasionally fail if there is another process running this script
        # which makes the directory after the check has been performed
        os.mkdir(definitions.AMPGEN_DIR)

    # We need a random number generator to decide which elements are test/train
    gen = np.random.default_rng()

    # Read the dataframe
    with uproot.open(path) as ag_f:
        dataframe = _ampgen_df(gen, ag_f["DalitzEventList"], sign)

    # Dump it
    with open(dump_path, "wb") as f:
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
