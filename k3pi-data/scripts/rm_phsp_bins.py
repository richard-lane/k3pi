"""
Remove phase space bin indices from dataframes

"""
import sys
import pickle
import pathlib
import argparse
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))

from lib_data import definitions


def main(args: argparse.Namespace) -> None:
    """Delete phsp bin columns from dataframes"""
    year, sign, magnetisation = args.year, args.sign, args.magnetisation

    dump_paths = [
        definitions.data_dump(path, year, sign, magnetisation)
        for path in definitions.data_files(year, magnetisation)
    ]

    col_header = "phsp bin"

    for path in tqdm(dump_paths):
        with open(path, "rb") as f:
            dataframe = pickle.load(f)

        if col_header not in dataframe:
            continue

        dataframe.drop(columns=col_header, inplace=True)

        with open(path, "wb") as f:
            pickle.dump(dataframe, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a column for phase space bin index to the DataFrames to `dumps/`"
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
