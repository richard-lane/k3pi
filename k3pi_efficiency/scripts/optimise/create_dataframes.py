"""
Create dataframes that will be used as input for the optimisation

NB this requires that the k3pi-data repo is cloned in the same
dir as k3pi_efficiency was, and that the particle gun and
AmpGen dataframes have been created

"""
import sys
import pickle
import pathlib
import argparse

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))

from lib_data import get


def main(
    sign: str,
):
    """
    Read the right dataframes, dump them to a sensible place

    """
    suffix = f"dataframe_{sign}.pkl"

    ampgen_df = get.ampgen(sign)
    pgun_df = get.particle_gun(sign, show_progress=True)

    with open(f"ampgen_{suffix}", "wb") as f:
        pickle.dump(ampgen_df, f)
    with open(f"pgun_{suffix}", "wb") as f:
        pickle.dump(pgun_df, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataframes that will be used as input when running BDT optimisation."
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="type of decay, CF or DCS",
    )

    args = parser.parse_args()
    main(args.sign)
