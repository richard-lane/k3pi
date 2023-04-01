"""
Create full LHCb MC dataframes

The MC files live on lxplus; this script should therefore be run on lxplus.

"""
import os
import time
import pickle
import argparse

import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import cuts, definitions, read, util


def _mc_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    # Convert the right branches into a dataframe
    start = time.time()
    dataframe = tree.arrays(read.branches("MC"), library="pd")
    dataframe = read.remove_refit(dataframe)
    read_time = time.time() - start

    util.ctau2lifetimes(dataframe)

    start = time.time()
    keep = cuts.mc_keep(dataframe)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    print(f"read/cut : {read_time:.3f}/{cut_time:.3f}")

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
