"""
Create dataframes for real data

There are lots of files - we will save each of these are their own DataFrame, for now

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import os
import time
import pickle
import pathlib
import argparse
from multiprocessing import get_context
import pandas as pd
from tqdm import tqdm
import uproot

from lib_data import definitions, cuts, util, read, training_vars


def _real_df(tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    """
    # Convert the right branches into a dataframe
    start = time.time()
    dataframe = tree.arrays(read.branches("data"), library="pd")
    dataframe = read.remove_refit(dataframe)
    training_vars.add_slowpi_pt_col(dataframe)
    read_time = time.time() - start

    util.ctau2lifetimes(dataframe)

    start = time.time()
    keep = cuts.data_keep(dataframe)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    # Rename branch -> column names
    util.rename_cols(dataframe)

    print(f"read/cut : {read_time:.3f}/{cut_time:.3f}")

    return dataframe


def _create_dump(
    data_path: pathlib.Path, dump_path: pathlib.Path, tree_name: str
) -> None:
    """
    Create a pickle dump of a dataframe

    """
    if dump_path.is_file():
        return

    with uproot.open(data_path) as data_f:
        # Create the dataframe
        dataframe = _real_df(data_f[tree_name])

    # Dump it
    with open(dump_path, "wb") as dump_f:
        pickle.dump(dataframe, dump_f)


def main(args: argparse.Namespace) -> None:
    """Create a DataFrame holding real data info"""
    # We might only want to iterate over a certain number of files
    n_files = args.n
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    data_paths = definitions.data_files(year, magnetisation)[:n_files]

    # If the dir doesnt exist, create it
    if not definitions.DATA_DIR.is_dir():
        os.mkdir(definitions.DATA_DIR)
    if not definitions.data_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.data_dir(year, sign, magnetisation))

    dump_paths = [
        definitions.data_dump(path, year, sign, magnetisation) for path in data_paths
    ]
    # Ugly - also have a list of tree names so i can use a starmap to iterate over both in parallel
    tree_names = [definitions.data_tree(sign) for _ in dump_paths]

    with get_context("spawn").Pool(args.n_procs) as pool:
        tqdm(
            pool.starmap(_create_dump, zip(data_paths, dump_paths, tree_names)),
            total=len(dump_paths),
        )
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "year",
        type=str,
        choices={"2016", "2017", "2018"},
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
