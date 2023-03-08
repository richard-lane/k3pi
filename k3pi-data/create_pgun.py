"""
Create particle gun dataframes

There are lots of particle gun files - we will save each of these are their own DataFrame, for now

The particle gun files live on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import pathlib
import argparse
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import definitions
from lib_data import cuts
from lib_data import training_vars
from lib_data import util
from lib_data import get


def _pgun_df(gen: np.random.Generator, data_tree, hlt_tree, mc_tree) -> pd.DataFrame:
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

    # MC correction stuff
    util.add_d0_momentum(dataframe, data_tree, keep)
    util.add_d0_eta(dataframe, data_tree, keep)

    util.add_train_column(gen, dataframe)

    # Also find how many generated events there are + return
    return dataframe, mc_tree.num_entries


def main(sign: str, n_files: int) -> None:
    """Create a DataFrame holding AmpGen momenta"""
    # Check if this exists first, to avoid confusion
    # In case we resume generating later + accidentally overwrite the file
    n_gen_file = get.pgun_n_gen_file(sign)
    assert not n_gen_file.is_file()

    # If the dir doesnt exist, create it
    if not definitions.PGUN_DIR.is_dir():
        os.mkdir(definitions.PGUN_DIR)
    if not definitions.pgun_dir(sign).is_dir():
        os.mkdir(definitions.pgun_dir(sign))

    # Generator for train/test RNG
    gen = np.random.default_rng()

    generated = []

    # Iterate over input files
    for path in tqdm(definitions.pgun_filepaths(sign)[:n_files]):
        dump_path = definitions.pgun_dump_fromfile(path, sign)

        # If the dump already exists, do nothing
        if dump_path.is_file():
            continue

        # Otherwise read the right trees
        data_path = pathlib.Path(path) / "pGun_TRACK.root"
        hlt_path = pathlib.Path(path) / "Hlt1TrackMVA.root"

        with uproot.open(str(data_path)) as data_f, uproot.open(str(hlt_path)) as hlt_f:
            data_tree = data_f["Dstp2D0pi/DecayTree"]
            hlt_tree = hlt_f["DecayTree"]
            mc_tree = data_f["MCDstp2D0pi/MCDecayTree"]

            # Create the dataframe
            dataframe, n_gen = _pgun_df(gen, data_tree, hlt_tree, mc_tree)

            generated.append(n_gen)
            print(n_gen)

        # Dump it
        with open(dump_path, "wb") as dump_f:
            pickle.dump(dataframe, dump_f)

    print("writing info about n gen to file")
    with open(str(n_gen_file), "w", encoding="utf8") as gen_f:
        gen_f.write("\n".join(str(n) for n in generated))


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
