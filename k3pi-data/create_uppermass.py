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
from multiprocessing import get_context

import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import cuts, definitions, read, util, training_vars


def _uppermass_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe information used for the classification

    """
    # Convert the right branches into a dataframe
    start = time.time()
    dataframe = tree.arrays(read.branches("data"), library="pd")
    dataframe = read.remove_refit(dataframe)
    training_vars.add_slowpi_pt_col(dataframe)
    read_time = time.time() - start

    util.ctau2lifetimes(dataframe)

    start = time.time()
    keep = cuts.uppermass_keep(dataframe)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    # Rename branch -> column names
    util.rename_cols(dataframe)

    dataframe = cuts.cands_cut(dataframe)
    dataframe = cuts.ipchi2_cut(dataframe)

    print(f"read/cut : {read_time:.3f}/{cut_time:.3f}")

    # Train test
    # More data for training than testing, since we only need ~12k evts for testing
    # since that's the expected stats
    util.add_train_column(gen, dataframe, train_fraction=0.85)

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

    # If the dir doesnt exist, create it
    if not definitions.UPPERMASS_DIR.is_dir():
        os.mkdir(definitions.UPPERMASS_DIR)
    if not definitions.uppermass_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.uppermass_dir(year, sign, magnetisation))

    dump_paths = [
        definitions.uppermass_dump(path, year, sign, magnetisation)
        for path in data_paths
    ][:n_files]

    # Iterable of the tree names so we can iterate over them in parallel with starmap
    tree_names = (definitions.data_tree(sign) for _ in range(n_files))

    with get_context("spawn").Pool(args.n_procs) as pool:
        pool.starmap(_create_dump, zip(data_paths, dump_paths, tree_names))
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
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
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="number of files to process; defaults to all of them",
    )

    parser.add_argument("--n_procs", type=int, default=2, help="number of processes")

    main(parser.parse_args())
