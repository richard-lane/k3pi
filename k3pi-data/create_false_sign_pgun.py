"""
Create dataframes for false sign particle gun - i.e. RS phase space events where the slow pion is
the wrong charge.
This simulates events where the D has undergone mixing

There are lots of particle gun files - we will save each of these are their own DataFrame, for now

The particle gun files live on lxplus; this script should therefore be run on lxplus.

"""
import os
import time
import pickle
import argparse
import uproot
import pandas as pd
from tqdm import tqdm

from lib_data import definitions, cuts, util, read, training_vars


def _false_sign_df(data_tree, hlt_tree) -> pd.DataFrame:
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

    # Perform cuts
    start = time.time()
    keep = cuts.pgun_keep(dataframe, hlt_df)
    dataframe = dataframe[keep]
    cut_time = time.time() - start

    # Swap momenta, which needs to be done
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

    print(f"read/cut/shuffle: {read_time:.3f}/{cut_time:.3f}{shuffle_time:.3f}")

    # Convert decay times from ctau to lifetimes
    util.ctau2lifetimes(dataframe)

    util.rename_cols(dataframe)

    return dataframe


def main() -> None:
    """Create a DataFrame holding false sign particle gun momenta"""
    # If the dir doesnt exist, create it
    if not definitions.FALSE_SIGN_DIR.is_dir():
        os.mkdir(definitions.FALSE_SIGN_DIR)

    # Iterate over input files
    for folder in tqdm(tuple(definitions.FALSE_SIGN_SOURCE_DIR.glob("*"))):
        # If the dump already exists, do nothing
        dump_path = definitions.false_sign_dump(folder.name)
        if dump_path.is_file():
            continue

        # Otherwise read the right trees
        data_path = folder / "pGun_TRACK.root"
        hlt_path = folder / "Hlt1TrackMVA.root"

        with uproot.open(data_path) as data_f, uproot.open(hlt_path) as hlt_f:
            data_tree = data_f["Dstp2D0pi/DecayTree"]
            hlt_tree = hlt_f["DecayTree"]

            # Create the dataframe
            dataframe = _false_sign_df(data_tree, hlt_tree)

        # Dump it
        with open(dump_path, "wb") as dump_f:
            pickle.dump(dataframe, dump_f)


if __name__ == "__main__":
    # No args since there's only one dir of false sign pgun stuff
    # but keep this in anyway
    parser = argparse.ArgumentParser(
        description="Dump 'false sign' pandas DataFrames to `dumps/`; i.e. RS evts that have undergone mixing"
    )

    main()
