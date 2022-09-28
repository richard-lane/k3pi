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
from lib_data import util
from lib_data import corrections


def _add_momenta(df: pd.DataFrame, data_tree, keep: np.ndarray) -> None:
    """
    Read momenta into the dataframe, in place

    Reads from the tree, applies the keep mask, adds columns to df

    """
    suffices = "PX", "PY", "PZ", "PE"
    branches = (
        *(f"Dst_ReFit_D0_Kplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_1_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_0_{s}" for s in suffices),
    )

    for branch, column in zip(branches, definitions.MOMENTUM_COLUMNS):
        # Take the first (best fit) value for each momentum
        df[column] = data_tree[branch].array()[:, 0][keep]


def _pgun_df(gen: np.random.Generator, data_tree, hlt_tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    df = pd.DataFrame()

    # Mask to perform straight cuts
    keep = cuts.sanity_keep(data_tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.hlt_keep_pgun(hlt_tree)

    # Combine with the background category mask
    keep &= cuts.bkgcat(data_tree)

    # Read momenta into the dataframe
    _add_momenta(df, data_tree, keep)

    # Read times into dataframe
    # 0.3 to convert from ctau to ps
    # 0.41 to convert from ps to D lifetimes
    # Take the first (best fit) value from each
    df["time"] = data_tree["Dst_ReFit_D0_ctau"].array()[:, 0][keep] / (0.3 * 0.41)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = data_tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]

    util.add_train_column(gen, df)

    return df


def main(sign: str) -> None:
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
    for folder in tqdm(tuple(source_dir.glob("*"))):
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

    args = parser.parse_args()

    main(args.sign)
