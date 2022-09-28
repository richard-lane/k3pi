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


def _add_momenta(df: pd.DataFrame, tree, keep: np.ndarray) -> None:
    """
    Read momenta into the dataframe, in place

    Reads from the tree, applies the keep mask, adds columns to df

    """
    suffices = "PX", "PY", "PZ", "PE"
    branches = (
        *(f"Dst_ReFit_D0_Kplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_0_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_1_{s}" for s in suffices),
    )

    for branch, column in zip(branches, definitions.MOMENTUM_COLUMNS):
        # Take the first (best fit) value for each momentum
        df[column] = tree[branch].array()[:, 0][keep]


def _mc_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    Provide the data + HLT information trees separately, since they live in different files

    """
    df = pd.DataFrame()

    # Mask to perform straight cuts
    keep = cuts.sanity_keep(tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.trigger_keep(tree)

    # Combine with the background category mask
    keep &= cuts.bkgcat(tree)

    # Read momenta into the dataframe
    _add_momenta(df, tree, keep)

    # Read training variables used for the classifier
    for branch, array in zip(
        training_vars.training_var_names(), training_vars.training_var_functions()
    ):
        df[branch] = array(tree)[keep]

    # Read times into dataframe
    # 0.3 to convert from ctau to ps
    # 0.41 to convert from ps to D lifetimes
    # Take the first (best fit) value from each
    df["time"] = tree["Dst_ReFit_D0_ctau"].array()[:, 0][keep] / (0.3 * 0.41)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]

    df["D0 mass"] = tree["Dst_ReFit_D0_M"].array()[:, 0][keep]
    df["D* mass"] = tree["Dst_ReFit_M"].array()[:, 0][keep]

    # track/SPD for event multiplicity reweighting
    corrections.add_multiplicity_columns(tree, df, keep)

    # Train test
    util.add_train_column(gen, df)

    return df


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
