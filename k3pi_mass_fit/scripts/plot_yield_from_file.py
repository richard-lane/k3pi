"""
Plot yields from a file on a weird histogram type thing

"""
import sys
import pickle
import pathlib
import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import ipchi2_fit
from libFit import util as mass_util


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bins: List[int],
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
    sec_correction: bool,
    scaled_yield: bool,
    integrated: bool,
):
    """
    From a file of yields, time bins etc., plot the
    RS + WS counts in each bin, their errors and the ratio

    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    hist_kw = {"histtype": "step"}

    if integrated:
        # Append a None to the list as this is the secret special value
        # that means we want the phsp integrated stuff
        phsp_bins += [None]

    # Get the secondary fractions if we need to
    if sec_correction:
        rs_sec_frac = ipchi2_fit.sec_fracs("cf")
        ws_sec_frac = ipchi2_fit.sec_fracs("dcs")

    for phsp_bin in phsp_bins:
        yield_file_path = mass_util.yield_file(
            year,
            magnetisation,
            phsp_bin,
            bdt_cut,
            efficiency,
            alt_bkg,
        )
        if scaled_yield:
            yield_file_path = mass_util.scaled_yield_file_path(yield_file_path)

        assert yield_file_path.exists(), yield_file_path

        # Get time bins, yields and errors
        time_bins, rs_yields, rs_errs, ws_yields, ws_errs = mass_util.read_yield(
            yield_file_path
        )

        # Do secondary fraction correction if we need to
        if sec_correction:
            rs_yields = ipchi2_fit.correct(rs_yields, rs_sec_frac)
            rs_errs = ipchi2_fit.correct(rs_errs, rs_sec_frac)

            ws_yields = ipchi2_fit.correct(ws_yields, ws_sec_frac)
            ws_errs = ipchi2_fit.correct(ws_errs, ws_sec_frac)

        # Get densities
        widths = time_bins[1:] - time_bins[:-1]
        centres = (time_bins[1:] + time_bins[:-1]) / 2

        rs_density = rs_yields / widths
        rs_density_err = rs_errs / widths

        ws_density = ws_yields / widths
        ws_density_err = ws_errs / widths

        ratio = ws_density / rs_density
        ratio_err = ratio * np.sqrt(
            (rs_density_err / rs_density) ** 2 + (ws_density_err / ws_density) ** 2
        )

        axes[0].hist(time_bins[:-1], bins=time_bins, weights=rs_density, **hist_kw)
        axes[1].hist(time_bins[:-1], bins=time_bins, weights=ws_density, **hist_kw)
        axes[2].errorbar(
            centres,
            ratio,
            xerr=widths / 2,
            yerr=ratio_err,
            label=f"Phsp bin {phsp_bin}" if phsp_bin is not None else "Phsp Integrated",
            fmt="+",
        )

    axes[2].legend()
    axes[2].set_xlabel(r"t/$\tau$")

    axes[0].set_ylabel("RS")
    axes[1].set_ylabel("WS")
    axes[2].set_ylabel(r"$\frac{WS}{RS}$")

    # axes[0].set_ylim([0.0, 3e6])
    # axes[1].set_ylim([0.0, 10000])
    # axes[2].set_ylim([0.0022, 0.0040])

    fig.tight_layout()

    path = f"yields_{year}_{magnetisation}_{bdt_cut=}_{efficiency=}_{alt_bkg=}_{sec_correction=}.png"
    if scaled_yield:
        path = f"scaled_{path}"
    print(f"plotting {path}")
    fig.savefig(path)

    with open(path, "wb") as f:
        pickle.dump((fig, axes), f"plot_pkls/{path}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mass fit plots")
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")
    parser.add_argument(
        "--efficiency", action="store_true", help="Correct for the detector efficiency"
    )
    parser.add_argument(
        "--alt_bkg", action="store_true", help="Use the alternate background model"
    )
    parser.add_argument(
        "phsp_bins", type=int, choices=range(4), help="Phase space bin index", nargs="*"
    )
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Also plot the phsp bin integrated stuff",
    )
    parser.add_argument(
        "--scaled_yield",
        action="store_true",
        help="Whether to use the yield file that has been scaled by luminosities",
    )
    parser.add_argument(
        "--sec_correction",
        action="store_true",
        help="Whether to use the yield file after secondary fraction correction",
    )

    main(**vars(parser.parse_args()))
