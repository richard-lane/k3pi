"""
Create particle gun dataframes

There are lots of particle gun files - we will save each of these are their own DataFrame, for now

The particle gun files live on lxplus; this script should therefore be run on lxplus.

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


def _pgun_df(gen: np.random.Generator, data_tree, hlt_tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    dataframe = pd.DataFrame()
    keep = cuts.pgun_keep(data_tree, hlt_tree)

    # Read momenta into the dataframe
    util.add_momenta(dataframe, data_tree, keep)
    # for reasons that make no sense to me, the particle gun definitions
    # for two of the pions are swapped
    # Deal with that here
    pi1_cols = definitions.MOMENTUM_COLUMNS[8:12]
    pi2_cols = definitions.MOMENTUM_COLUMNS[12:16]
    dataframe.rename(
        columns=dict(zip(pi1_cols, pi2_cols), **dict(zip(pi2_cols, pi1_cols))),
        inplace=True,
    )
    cols = dataframe.columns.tolist()
    cols = [*cols[0:4], *cols[8:12], *cols[4:8], *cols[12:]]
    dataframe = dataframe[cols]

    util.add_refit_times(dataframe, data_tree, keep)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    training_vars.add_vars(dataframe, data_tree, keep)
    util.add_k_id(dataframe, data_tree, keep)
    util.add_masses(dataframe, data_tree, keep)

    util.add_train_column(gen, dataframe)

    return dataframe


def main(sign: str, n_files: int) -> None:
    """Create a DataFrame holding AmpGen momenta"""
    # If the dir doesnt exist, create it
    if not definitions.PGUN_DIR.is_dir():
        os.mkdir(definitions.PGUN_DIR)
    if not definitions.pgun_dir(sign).is_dir():
        os.mkdir(definitions.pgun_dir(sign))

    source_dir = (
        definitions.RS_PGUN_SOURCE_DIR
        if sign == "cf"
        else definitions.WS_PGUN_SOURCE_DIR
    )

    # Keep track of which folders broke - this might be expected
    broken_folders = []

    # Generator for train/test RNG
    gen = np.random.default_rng()

    # Iterate over input files
    for folder in tqdm(tuple(source_dir.glob("*"))[:n_files]):
        # If the dump already exists, do nothing
        dump_path = definitions.pgun_dump(sign, folder.name)
        if dump_path.is_file():
            continue

        # Otherwise read the right trees
        data_path = folder / "pGun_TRACK.root"
        hlt_path = folder / "Hlt1TrackMVA.root"
        try:
            with uproot.open(data_path) as data_f, uproot.open(hlt_path) as hlt_f:
                data_tree = data_f["Dstp2D0pi/DecayTree"]
                hlt_tree = hlt_f["DecayTree"]

                # Create the dataframe
                dataframe = _pgun_df(gen, data_tree, hlt_tree)

            # Dump it
            with open(dump_path, "wb") as dump_f:
                pickle.dump(dataframe, dump_f)

        except FileNotFoundError:
            broken_folders.append(str(folder))
            continue

    if broken_folders:
        print(f"Failed to read from dirs:\n\t{broken_folders}")
        print(
            "This may be expected, e.g. there may be already merged files also in the dir"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="number of files to process; defaults to all of them",
    )

    args = parser.parse_args()

    main(args.sign, args.n)
