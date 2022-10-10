"""
Create dataframes for false sign particle gun - i.e. RS phase space events where the slow pion is
the wrong charge.
This simulates events where the D has undergone mixing

There are lots of particle gun files - we will save each of these are their own DataFrame, for now

The particle gun files live on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import argparse
import uproot
import pandas as pd
from tqdm import tqdm

from lib_data import definitions, cuts, util


def _false_sign_df(data_tree, hlt_tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    dataframe = pd.DataFrame()

    keep = cuts.pgun_keep(data_tree, hlt_tree)

    util.add_momenta(dataframe, data_tree, keep)

    util.add_refit_times(dataframe, data_tree, keep)

    util.add_k_id(dataframe, data_tree, keep)

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
    # No args at this stage, keep this in though for consistency with the other scripts
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.parse_args()

    main()
