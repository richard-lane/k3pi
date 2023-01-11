"""
Simultaneous and individual fits to RS and WS

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))

from libFit import pdfs, fit, util as mass_util, plotting, definitions
from lib_time_fit.definitions import TIME_BINS


def _separate_fit(
    count: np.ndarray,
    err: np.ndarray,
    bin_number: int,
    bins: np.ndarray,
    sign: str,
    plot_path: str,
    *,
    alt_bkg: bool,
    bdt_cut: bool,
    efficiency: bool,
):
    """
    Plot separate fits

    """
    fitter = (
        fit.alt_bkg_fit(
            count,
            bins,
            sign,
            bin_number,
            0.9 if sign == "cf" else 0.05,
            errors=err,
            bdt_cut=bdt_cut,
            efficiency=efficiency,
        )
        if alt_bkg
        else fit.binned_fit(
            count, bins, sign, bin_number, 0.9 if sign == "cf" else 0.05, errors=err
        )
    )

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(6, 8))

    if alt_bkg:
        plotting.alt_bkg_fit(
            (axes["A"], axes["B"]),
            count,
            err,
            bins,
            fitter.values,
            sign=sign,
            bdt_cut=False,
            efficiency=False,
        )
    else:
        plotting.mass_fit((axes["A"], axes["B"]), count, err, bins, fitter.values)

    axes["A"].legend()

    axes["B"].plot(pdfs.domain(), [1, 1], "r-")

    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()

    fig.tight_layout()
    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)


def _fit(
    rs_count: np.ndarray,
    ws_count: np.ndarray,
    rs_err: np.ndarray,
    ws_err: np.ndarray,
    bin_number: int,
    bins: np.ndarray,
    fit_dir: str,
    *,
    alt_bkg: bool,
    bdt_cut: bool,
    efficiency: bool,
) -> None:
    """
    Plot the fit

    """
    fitter = (
        fit.alt_simultaneous_fit(
            rs_count,
            ws_count,
            bins,
            bin_number,
            rs_err,
            ws_err,
            bdt_cut=bdt_cut,
            efficiency=efficiency,
        )
        if alt_bkg
        else fit.binned_simultaneous_fit(
            rs_count, ws_count, bins, bin_number, rs_err, ws_err
        )
    )
    params = fitter.values
    print(f"{fitter.valid=}", end="\t")
    print(f"{params[-2]} +- {fitter.errors[-2]}", end="\t")
    print(f"{params[-1]} +- {fitter.errors[-1]}")

    fig, _ = (
        plotting.alt_bkg_simul(
            rs_count,
            rs_err,
            ws_count,
            ws_err,
            bins,
            params,
            bdt_cut=False,
            efficiency=False,
        )
        if alt_bkg
        else plotting.simul_fits(
            rs_count,
            rs_err,
            ws_count,
            ws_err,
            bins,
            params,
        )
    )

    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()

    bkg_str = "_alt_bkg" if alt_bkg else ""
    plot_path = f"{fit_dir}fit_{bkg_str}{bin_number}.png"

    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)


def main(args: argparse.Namespace):
    """
    Do mass fits in each time bin without BDT cuts

    """
    if args.efficiency:
        assert args.bdt_cut

    bins = definitions.mass_bins(100)
    time_bins = np.array((-np.inf, *TIME_BINS[1:], np.inf))

    year, magnetisation = args.year, args.magnetisation

    dcs_counts, cf_counts, dcs_errs, cf_errs = mass_util.mass_counts(
        year,
        magnetisation,
        bins,
        time_bins,
        bdt_cut=args.bdt_cut,
        correct_efficiency=args.efficiency,
        phsp_bin=args.phsp_bin,
    )

    plot_dir = mass_util.plot_dir(args.bdt_cut, args.efficiency, args.phsp_bin)

    bkg_str = "alt_bkg_" if args.alt_bkg else ""

    for i, (dcs_count, cf_count, dcs_err, cf_err) in enumerate(
        zip(dcs_counts[1:-1], cf_counts[1:-1], dcs_errs[1:-1], cf_errs[1:-1])
    ):
        _fit(
            cf_count,
            dcs_count,
            cf_err,
            dcs_err,
            i,
            bins,
            plot_dir,
            alt_bkg=args.alt_bkg,
            bdt_cut=args.bdt_cut,
            efficiency=args.efficiency,
        )
        _separate_fit(
            cf_count,
            cf_err,
            i,
            bins,
            "cf",
            f"{plot_dir}{bkg_str}rs/{i}.png",
            alt_bkg=args.alt_bkg,
            bdt_cut=args.bdt_cut,
            efficiency=args.efficiency,
        )
        _separate_fit(
            dcs_count,
            dcs_err,
            i,
            bins,
            "dcs",
            f"{plot_dir}{bkg_str}ws/{i}.png",
            alt_bkg=args.alt_bkg,
            bdt_cut=args.bdt_cut,
            efficiency=args.efficiency,
        )


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
        help="Whether to attempt the fits with the alternate background model",
    )

    main(parser.parse_args())
