"""
Scale the yield file output from fits to data by
the corresponding luminosities
This luminosity can either be provided on the CLI
or calculated from the files on /eos/

"""
import sys
import pathlib
import argparse

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from libFit import util
from lib_data import definitions
from lib_data.util import corresponding_lumi


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bin: int,
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
    dcs_lumi: float,
    cf_lumi: float,
):
    """
    Do mass fits in each time bin

    """
    # Get the right yield file
    in_file = util.yield_file(
        year,
        magnetisation,
        phsp_bin,
        bdt_cut,
        efficiency,
        alt_bkg,
    )
    assert in_file.exists()

    # Get the luminosities if not provided on the CLI
    dcs_lumi = (
        dcs_lumi
        if dcs_lumi is not None
        else corresponding_lumi(
            definitions.data_dir(year, "dcs", magnetisation).glob("*.pkl")
        )
    )
    cf_lumi = (
        cf_lumi
        if cf_lumi is not None
        else corresponding_lumi(
            definitions.data_dir(year, "cf", magnetisation).glob("*.pkl")
        )
    )

    print(f"{dcs_lumi=}")
    print(f"{cf_lumi=}")

    # Write to a new file
    util.write_scaled_yield(in_file, dcs_lumi, cf_lumi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mass fit plots")
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
        "phsp_bin", type=int, choices=range(4), help="Phase space bin index"
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")
    parser.add_argument(
        "--efficiency", action="store_true", help="Correct for the detector efficiency"
    )
    parser.add_argument(
        "--alt_bkg",
        action="store_true",
        help="Whether to fit with the alternate bkg model",
    )
    parser.add_argument(
        "--dcs_lumi",
        type=float,
        help="Luminosity used to create the DCS dfs. If not specified, will be calculated from the dumps on /eos/",
        default=None,
    )
    parser.add_argument(
        "--cf_lumi",
        type=float,
        help="Luminosity used to create the dfs. If not specified, will be calculated from the dumps on /eos/",
        default=None,
    )

    main(**vars(parser.parse_args()))
