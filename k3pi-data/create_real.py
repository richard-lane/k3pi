"""
Create dataframes for real data

There are lots of files - we will save each of these are their own DataFrame, for now

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import pathlib
import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import uproot

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
        *(f"Dst_ReFit_piplus_{s}" for s in suffices),
    )

    for branch, column in zip(
        branches,
        [
            *definitions.MOMENTUM_COLUMNS,
            *(f"slowpi_{s}" for s in ("Px", "Py", "Pz", "E")),
        ],
    ):
        # Take the first (best fit) value for each momentum
        df[column] = tree[branch].array()[:, 0][keep]


def _keep(tree):
    """mask of evts to keep"""
    # Mask to perform straight cuts
    keep = cuts.sanity_keep(tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.trigger_keep(tree)

    return keep


def _real_df(tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    """
    df = pd.DataFrame()

    keep = _keep(tree)

    # Read momenta into the dataframe
    _add_momenta(df, tree, keep)

    # Read times into dataframe
    # 0.3 to convert from ctau to ps
    # 0.41 to convert from ps to D lifetimes
    # Take the first (best fit) value from each
    df["time"] = tree["Dst_ReFit_D0_ctau"].array()[:, 0][keep] / (0.3 * 0.41)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]
    for branch, array in zip(
        training_vars.training_var_names(), training_vars.training_var_functions()
    ):
        df[branch] = array(tree)[keep]

    # D, D* masses
    df["D0 mass"] = tree["Dst_ReFit_D0_M"].array()[:, 0][keep]
    df["D* mass"] = tree["Dst_ReFit_M"].array()[:, 0][keep]

    # Slow pi ID
    df["slow pi ID"] = tree["Dst_ReFit_piplus_ID"].array()[:, 0][keep]

    # track/SPD for event multiplicity reweighting
    corrections.add_multiplicity_columns(tree, df, keep)

    return df


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
    print(f"dumping {dump_path}")
    with open(dump_path, "wb") as dump_f:
        pickle.dump(dataframe, dump_f)


def _n(tree):
    """number of evts in signal region in a tree"""
    keep = _keep(tree)
    delta_m = (
        tree["Dst_ReFit_M"].array()[:, 0][keep]
        - tree["Dst_ReFit_D0_M"].array()[:, 0][keep]
    )

    lo, hi = 144, 147

    return np.sum((144 < delta_m) & (delta_m < 147))


def _n_region(paths, sign):
    """Count the total number of evts in signal region"""
    total = 0
    lumi = 0
    tree_name = definitions.data_tree(sign)

    for path in tqdm(paths):
        l = util.luminosity(path)
        try:
            with uproot.open(path) as f:
                n = _n(f[tree_name])

                total += n
                lumi += l
        except KeyboardInterrupt:
            print(total * 620.3547250896617 / lumi)
            return total, lumi


def main(args: argparse.Namespace) -> None:
    """Create a DataFrame holding real data info"""
    year, sign, magnetisation = args.year, args.sign, args.magnetisation
    data_paths = definitions.data_files(year, magnetisation)

    if args.print_lumi:
        print(f"total luminosity: {util.total_luminosity(data_paths)}")
        return

    if args.print_count:
        print(f"total number in region, lumi: {_n_region(data_paths, sign)}")
        return

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

    with Pool(processes=8) as pool:
        tqdm(
            pool.starmap(_create_dump, zip(data_paths, dump_paths, tree_names)),
            total=len(dump_paths),
        )


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

    # TODO use a subparser to make args conditional on this
    parser.add_argument(
        "--print_count",
        action="store_true",
        help="Iterate over all files, print total number of events in a region.",
    )

    main(parser.parse_args())
