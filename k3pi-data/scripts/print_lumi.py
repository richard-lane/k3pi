"""
Iterate over real data analysis production files and add up
their luminosities

"""
import sys
import pathlib
import argparse

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, util


def main(args):
    """
    Open the files and add the luminosities up

    """
    year, magnetisation = args.year, args.magnetisation
    data_paths = definitions.data_files(year, magnetisation)

    print(f"total luminosity: {util.total_luminosity(data_paths)}")


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

    main(parser.parse_args())
