"""
Iterate over real data analysis production files, counting how
many events are in the signal region

"""
import sys
import pathlib
import argparse
from tqdm import tqdm
import numpy as np
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, cuts, util


def _n(tree):
    """number of evts in signal region in a tree"""
    keep = cuts.data_keep(tree)
    delta_m = cuts.dst_mass(tree)[keep] - cuts.d0_mass(tree)[keep]

    lo_mass, hi_mass = 144, 147

    return np.sum((lo_mass < delta_m) & (delta_m < hi_mass))


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
            # If we cancel early, scale the number of evts encountered in signal
            # region by the total expected/encountered luminosities
            # to give an estimate of the total number in signal region
            total_expected_lumi = 620.3547250896617
            print(f"stopped early; estimate:\n\t{total * total_expected_lumi / lumi}")
            return total, lumi

    return total, lumi


def main(args):
    """
    Open each file, count the number in the signal region

    """
    year, sign, magnetisation = args.year, args.sign, args.magnetisation
    data_paths = definitions.data_files(year, magnetisation)
    print(f"total number in region, lumi: {_n_region(data_paths, sign)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the number of evts in the signal region"
    )

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

    main(parser.parse_args())
