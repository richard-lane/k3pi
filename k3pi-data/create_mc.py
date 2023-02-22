"""
Create full LHCb MC dataframes

The MC files live on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import argparse
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import definitions
from lib_data import cuts
from lib_data import training_vars
from lib_data import util
from lib_data import corrections


def _mc_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    dataframe = pd.DataFrame()

    # Mask to perform straight cuts
    keep = cuts.simulation_keep(tree)

    # Read momenta into the dataframe
    util.add_momenta(dataframe, tree, keep)
    util.add_refit_times(dataframe, tree, keep)

    # Read training variables used for the classifier
    training_vars.add_vars(dataframe, tree, keep)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    util.add_k_id(dataframe, tree, keep)
    util.add_masses(dataframe, tree, keep)

    # MC correction stuff
    corrections.add_multiplicity_columns(tree, dataframe, keep)
    util.add_d0_momentum(dataframe, tree, keep)
    util.add_d0_eta(dataframe, tree, keep)

    # Train test
    util.add_train_column(gen, dataframe)

    return dataframe


def main(year: str, sign: str, magnetisation: str) -> None:
    """Create a DataFrame holding MC momenta"""
    # If the dir doesnt exist, create it
    if not definitions.MC_DIR.is_dir():
        os.mkdir(definitions.MC_DIR)

    # If the dump already exists, do nothing
    dump_path = definitions.mc_dump(year, sign, magnetisation)
    if dump_path.is_file():
        print(f"{dump_path} already exists")
        return

    # RNG for train test
    gen = np.random.default_rng(seed=0)

    # Iterate over input files
    dfs = []
    for data_path in tqdm(definitions.mc_files(year, magnetisation, sign)):
        with uproot.open(data_path) as data_f:
            tree = data_f[definitions.data_tree(sign)]

            # Create the dataframe
            dfs.append(_mc_df(gen, tree))

    # Concatenate dataframes and dump
    with open(dump_path, "wb") as dump_f:
        pickle.dump(pd.concat(dfs, ignore_index=True), dump_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="data taking year",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation)
