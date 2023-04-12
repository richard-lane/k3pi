"""
Create particle gun dataframes

There are lots of particle gun files - we will save each of these are their own DataFrame, for now

The particle gun files live on lxplus; this script should therefore be run on lxplus.

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

from lib_data import cuts, definitions, get, util, read, training_vars


def _pgun_df(gen: np.random.Generator, data_tree, hlt_tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    # Convert the right branches into a dataframe
    start = time.time()
    dataframe = data_tree.arrays(read.branches("pgun"), library="pd")
    dataframe = read.remove_refit(dataframe)

    training_vars.add_slowpi_pt_col(dataframe)

    # Also convert the right HLT branches from the other file
    # into a dataframe
    hlt_df = hlt_tree.arrays(read.pgun_hlt_branches(), library="pd")

    read_time = time.time() - start

    # Convert decay times from ctau to lifetimes
    util.ctau2lifetimes(dataframe)

    # Perform cuts
    start = time.time()
    keep = cuts.pgun_keep(dataframe, hlt_df)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    # Swap momenta
    # for reasons that make no sense to me, the particle gun definitions
    # for two of the pions are swapped; deal with that here
    start = time.time()
    pi1_cols = definitions.MOMENTUM_COLUMNS[8:12]
    pi2_cols = definitions.MOMENTUM_COLUMNS[12:16]
    dataframe.rename(
        columns=dict(zip(pi1_cols, pi2_cols), **dict(zip(pi2_cols, pi1_cols))),
        inplace=True,
    )

    # Reorder the momentum columns to be consistent with the others
    cols = dataframe.columns.tolist()
    cols = [*cols[0:4], *cols[8:12], *cols[4:8], *cols[12:]]
    dataframe = dataframe[cols]
    shuffle_time = time.time() - start

    print(f"read/cut/shuffle : {read_time:.3f}/{cut_time:.3f}/{shuffle_time:.3f}")

    # Rename branch -> column names
    util.rename_cols(dataframe)

    # Add test/train column
    util.add_train_column(gen, dataframe)

    # Also find how many generated events there are + return
    return dataframe


def main(
    *, year: str, sign: str, magnetisation: str, n_files: int, verbose: bool
) -> None:
    """Create a DataFrame holding AmpGen momenta"""
    # Hopefully it'll be obvious if this is
    # writing the right info, if we pause generation and
    # resume later...
    n_gen_file = get.pgun_n_gen_file(year, sign, magnetisation)

    # If the dir doesnt exist, create it
    if not definitions.PGUN_DIR.is_dir():
        os.mkdir(definitions.PGUN_DIR)
    if not definitions.pgun_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.pgun_dir(year, sign, magnetisation))

    data_paths = definitions.pgun_data_prods(year, sign, magnetisation)[:n_files]
    hlt_paths = definitions.pgun_hlt_prods(year, sign, magnetisation)[:n_files]

    # Generator for train/test RNG
    gen = np.random.default_rng()

    generated = []
    # Iterate over input files
    for data_path, hlt_path in tqdm(zip(data_paths, hlt_paths)):
        dump_path = definitions.pgun_dump_fromfile(data_path, year, sign, magnetisation)

        # If the dump already exists, do nothing
        if dump_path.is_file():
            if verbose:
                print(f"{dump_path} exists")
            continue

        with uproot.open(str(data_path)) as data_f, uproot.open(str(hlt_path)) as hlt_f:
            data_tree = data_f["Dstp2D0pi/DecayTree"]
            hlt_tree = hlt_f["DecayTree"]

            # Create the dataframe
            dataframe = _pgun_df(gen, data_tree, hlt_tree)

            # Find also the number generated
            n_gen = data_f["MCDstp2D0pi/MCDecayTree"].num_entries
            generated.append(n_gen)

            if verbose:
                print(f"{len(dataframe)=}\t{n_gen=}", end="\t")

        # Dump it
        with open(dump_path, "wb") as dump_f:
            if verbose:
                print(f"dumping {dump_path}")
            pickle.dump(dataframe, dump_f)

    print("writing info about n gen to file")
    with open(str(n_gen_file), "a", encoding="utf8") as gen_f:
        gen_f.write("\n".join(str(n) for n in generated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018", "2016"},
        help="Year - for trigger information",
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
        help="Mag direction - for trigger information",
    )
    parser.add_argument(
        "-n",
        "--n_files",
        type=int,
        default=None,
        help="number of files to process; defaults to all of them",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    main(**vars(parser.parse_args()))
