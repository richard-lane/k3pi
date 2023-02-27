"""
Find what the background distribution should look like
by finding the invariant masses M(K3pi) and M(K3pi+pi_s)
where the pi_s comes from a different event

Store the arrays of masses as pickle dumps

"""
import os
import sys
import pickle
import pathlib
import argparse
from typing import Iterable
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, definitions, util
from lib_data.util import k_3pi
from lib_cuts.get import cut_dfs, classifier as get_clf
from libFit import bkg, pdfs


def _n_data_files(year: str, sign: str, magnetisation: str) -> int:
    """Count how many pickle dumped dataframes there are"""
    dir_ = definitions.data_dir(year, sign, magnetisation)
    return len(
        [name for name in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, name))]
    )


def _create_dumps(
    dump_dir: pathlib.Path,
    df_generator: Iterable[pd.DataFrame],
    n_dfs: int,
    n_repeats: int,
) -> None:
    """
    From a generator of dataframes, find arrays of M(K4pi) - M(K3pi) for each dataframe
    and pickle those that fall into a desired range

    """
    # delta M values we want to bother caching
    min_delta_m, max_delta_m = pdfs.domain()[0], 160

    # Get lists of daughter particle arrays separately
    # In case we want to do a larger number of rolls, which might
    # involve having to concatenate the arrays
    with tqdm(total=n_dfs * n_repeats) as pbar:
        for i, dataframe in enumerate(df_generator):
            dump_path = str(dump_dir / f"{i}.pkl")

            # List to append delta M arrays to
            delta_ms = []

            # Get the kinematic 4-vectors
            k, pi1, pi2, pi3 = k_3pi(dataframe)
            slowpi = np.row_stack(
                [dataframe[f"slowpi_{s}"] for s in definitions.MOMENTUM_SUFFICES]
            )
            d_mass = util.inv_mass(k, pi1, pi2, pi3)

            # Shuffle the slow pi momenta by 1, find the delta M, append to list
            for _ in range(n_repeats):
                slowpi = np.roll(slowpi, 1, axis=1)
                dst_mass = util.inv_mass(k, pi1, pi2, pi3, slowpi)

                pbar.update(1)

                delta_m = dst_mass - d_mass

                keep = (min_delta_m < delta_m) & (delta_m < max_delta_m)
                delta_ms.append(delta_m[keep])

            delta_ms = np.concatenate(delta_ms)

            # Pickle dump the concatenated list
            with open(dump_path, "wb") as dump_f:
                pickle.dump(delta_ms, dump_f)


def main(*, year: str, magnetisation: str, sign: str, bdt_cut: bool, n_repeats: int):
    """
    Get the invariant masses from the right dataframes,
    store them in a histogram and plot it

    """
    # Get directory for storing dumps
    dump_dir = bkg.dump_dir(year, magnetisation, sign, bdt_cut=bdt_cut)

    # Check the dump dir is currently empty or nonexistent, otherwise we might end up
    # with the same arrays twice
    if dump_dir.is_dir():
        try:
            os.rmdir(str(dump_dir))  # Will error if not empty

        except OSError as err:
            print("=" * 8, f"{dump_dir} not empty", "=" * 8, sep="\n", file=sys.stderr)
            raise err

    dump_dir.mkdir()
    print(dump_dir)

    # Get a generator of dataframes
    df_generator = get.data(year, sign, magnetisation)
    if bdt_cut:
        clf = get_clf(year, "dcs", magnetisation)
        df_generator = cut_dfs(df_generator, clf)

    # Create the dumps in the dir from this generator
    _create_dumps(
        dump_dir, df_generator, _n_data_files(year, sign, magnetisation), n_repeats
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create pickle dumps of arrays for the empirical bkg shape"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="data type",
    )
    parser.add_argument(
        "--bdt_cut",
        action="store_true",
        help="whether to BDT cut the dataframes before finding bkg",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        help="how many times to roll the slowpis in each dataframe",
        default=100,
    )

    main(**vars(parser.parse_args()))
