"""
Iterate over the real data dumps that we've created and add up their luminosities

"""
import sys
import pathlib
import argparse

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, util


def main(*, year: str, magnetisation: str):
    """
    Open the files and add the luminosities up

    """
    cf_data_paths = definitions.data_dir(year, "cf", magnetisation).glob("*.pkl")
    dcs_data_paths = definitions.data_dir(year, "dcs", magnetisation).glob("*.pkl")

    print(f"CF dump luminosity: {util.corresponding_lumi(cf_data_paths)}")
    print(f"DCS dump luminosity: {util.corresponding_lumi(dcs_data_paths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the total luminosity in all the data files"
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

    main(**vars(parser.parse_args()))
