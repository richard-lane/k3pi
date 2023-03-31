"""
Create dataframes for the upper mass sideband of real data
Used when training the sig/bkg classifier

There are lots of files - we will save each of these are their own DataFrame, for now

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import os
import time
import pickle
import pathlib
import argparse

import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import cuts, definitions, read, util


def _uppermass_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe information used for the classification

    """
    # Convert the right branches into a dataframe
    start = time.time()
    dataframe = tree.arrays(read.branches("data"), library="pd")
    read_time = time.time() - start

    start = time.time()
    keep = cuts.data_keep(dataframe)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    print(f"read/cut : {read_time:.3f}/{cut_time:.3f}")

    # Train test
    util.add_train_column(gen, dataframe)

    return dataframe


def _create_dump(
    data_path: pathlib.Path, dump_path: pathlib.Path, tree_name: str
) -> None:
    """
    Create a pickle dump of a dataframe

    """
    # If the dump already exists, do nothing
    if dump_path.is_file():
        return

    # Create a new random generator every time
    # This isn't very good, but also it isn't a disaster
    # As long as the seed is actually random
    # TODO seed with pid, time, etc
    gen = np.random.default_rng()

    with uproot.open(data_path) as data_f:
        # Create the dataframe
        dataframe = _uppermass_df(gen, data_f[tree_name])

        # Add also a column for the luminosity
        # Do this by adding a column of zeros and then filling the first
        # entry with the required luminosity
        dataframe["luminosity"] = np.zeros(len(dataframe))
        dataframe.loc[0, "luminosity"] = util.luminosity(data_path)

    # Dump it
    print(f"dumping {dump_path}")
    with open(dump_path, "wb") as dump_f:
        pickle.dump(dataframe, dump_f)


def main(args: argparse.Namespace) -> None:
    """
    Create a DataFrame holding real data info from the upper mass sideband

    Used as a background sample

    """
    # We might only want to iterate over a certain number of files
    n_files = args.n
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    data_paths = definitions.data_files(year, magnetisation)[:n_files]

    if args.print_lumi:
        print(f"total luminosity: {util.total_luminosity(data_paths)}")
        return

    # If the dir doesnt exist, create it
    if not definitions.UPPERMASS_DIR.is_dir():
        os.mkdir(definitions.UPPERMASS_DIR)
    if not definitions.uppermass_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.uppermass_dir(year, sign, magnetisation))

    dump_paths = [
        definitions.uppermass_dump(path, year, sign, magnetisation)
        for path in data_paths
    ][:n_files]

    for data_path, dump_path in tqdm(zip(data_paths, dump_paths)):
        _create_dump(data_path, dump_path, definitions.data_tree(sign))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
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
        choices={"magdown"},
        help="magnetisation direction",
    )

    # TODO use a subparser to make args conditional on this
    parser.add_argument(
        "--print_lumi",
        action="store_true",
        help="Iterate over all files, print total luminosity and exit.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="number of files to process; defaults to all of them",
    )

    parser.add_argument("--n_procs", type=int, default=2, help="number of processes")

    main(parser.parse_args())
