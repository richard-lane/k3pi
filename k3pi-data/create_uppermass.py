"""
Create dataframes for the upper mass sideband of real data
Used when training the sig/bkg classifier

There are lots of files - we will save each of these are their own DataFrame, for now

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import pathlib
import argparse
from multiprocessing import Pool
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import definitions
from lib_data import cuts
from lib_data import training_vars
from lib_data import util


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


def _bkg_keep(d_mass: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """
    Mask of events to keep after background mass cuts

    """
    # Keep points far enough away from the nominal mass
    d_mass_width = 24
    d_mass_range = (
        definitions.D0_MASS_MEV - d_mass_width,
        definitions.D0_MASS_MEV + d_mass_width,
    )
    d_mass_mask = (d_mass < d_mass_range[0]) | (d_mass_range[1] < d_mass)

    # AND within the delta M upper mass sideband
    delta_m_mask = (152 < delta_m) & (delta_m < 157)

    return d_mass_mask & delta_m_mask


def _uppermass_df(gen: np.random.Generator, tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe information used for the classification

    """
    df = pd.DataFrame()

    # Mask to perform straight cuts
    # Usually we would do the "sanity cuts" here -
    # these are D0, delta M and ipchi2 cuts
    # In this case we just want the ipchi2
    keep = cuts._ipchi2(tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.trigger_keep(tree)

    # Keep only events in the upper mass sideband
    d_mass = tree["Dst_ReFit_D0_M"].array()[:, 0]
    dst_mass = tree["Dst_ReFit_M"].array()[:, 0]
    keep &= _bkg_keep(d_mass, dst_mass - d_mass)

    # Store the D masses
    df["D0 mass"] = d_mass[keep]
    df["D* mass"] = dst_mass[keep]

    # Read other things that we want to keep for training the signal cut BDT
    for branch, array in zip(
        training_vars.training_var_names(), training_vars.training_var_functions()
    ):
        df[branch] = array(tree)[keep]

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]
    df["slow pi ID"] = tree["Dst_ReFit_piplus_ID"].array()[:, 0][keep]

    # Read times into dataframe
    # 0.3 to convert from ctau to ps
    # 0.41 to convert from ps to D lifetimes
    # Take the first (best fit) value from each
    df["time"] = tree["Dst_ReFit_D0_ctau"].array()[:, 0][keep] / (0.3 * 0.41)

    # Also want momenta
    _add_momenta(df, tree, keep)

    # Train test
    util.add_train_column(gen, df)

    return df


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
    year, sign, magnetisation = args.year, args.sign, args.magnetisation
    data_paths = definitions.data_files(year, magnetisation)

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
    ]

    # Ugly - also have a list of tree names so i can use a starmap
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

    main(parser.parse_args())
