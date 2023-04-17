"""
Find the fraction of events that have multiple candidates

"""
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, cuts



def main(*, year: str, sign: str, magnetisation: str):
    """
    from our dataframes count how many candidates there are, how many events and how many
    events have multiple cands

    """
    n_cands = 0
    n_evts = 0
    n_evts_with_multiple_cands = 0

    # Count n cands after the ipchi2 cut
    for dataframe in tqdm(cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation))):
        n_cands += len(dataframe)

        cands_arr = dataframe["nCandidate"]
        n_evts += np.sum(cands_arr == 0)
        n_evts_with_multiple_cands += np.sum(cands_arr == 1)

    print(f"{n_cands=:,}\t{n_evts=:,}\t{n_evts_with_multiple_cands=:,}")
    print(f"Frac of evts with multiple cands: {100 * n_evts_with_multiple_cands / n_evts:.3f} %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find how many candidates come from the same event, and plot something maybe"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
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
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
