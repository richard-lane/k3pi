"""
Simultaneous and individual fits to RS and WS

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))

from lib_data import get
from lib_cuts.get import classifier as get_clf, cut_dfs
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_util import wts_generator
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
    try:
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
    except pdfs.ZeroCountsError as exc:
        print("time bin:", bin_number, repr(exc))
        return

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

    fig.suptitle(
        f"Nsig={fitter.values[0]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[0]:,.2f}"
        f"\tNbkg={fitter.values[1]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[1]:,.2f}"
        f"\t{fitter.valid=}"
    )
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
    try:
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
    except pdfs.ZeroCountsError as err:
        print("time bin:", bin_number, repr(err))
        return

    params = fitter.values
    print(f"{fitter.valid=}", end="\t")
    print(f"{params[-2]} +- {fitter.errors[-2]}", end="\t")
    print(f"{params[-1]} +- {fitter.errors[-1]}")

    fig, axes = (
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

    axes["A"].set_title(
        f"Nsig={fitter.values[0]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[0]:,.2f}"
        f"\tNbkg={fitter.values[1]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[1]:,.2f}"
    )
    axes["B"].set_title(
        f"Nsig={fitter.values[2]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[2]:,.2f}"
        f"\tNbkg={fitter.values[3]:,.2f}"
        r"$\pm$"
        f"{fitter.errors[3]:,.2f}"
    )
    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()
    fig.tight_layout()

    bkg_str = "_alt_bkg" if alt_bkg else ""
    plot_path = f"{fit_dir}fit_{bkg_str}{bin_number}.png"

    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)


def _dataframes(
    year: str,
    magnetisation: str,
    phsp_bin: int,
    sign: str,
    low_t: float,
    high_t: float,
    bdt_clf,
    bdt_cut,
):
    """
    The right dataframes

    """
    return cut_dfs(
        get.binned_generator(
            get.time_binned_generator(
                get.data(year, sign, magnetisation), low_t, high_t
            ),
            phsp_bin,
        ),
        bdt_clf,
        perform_cut=bdt_cut,
    )


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bin: int,
    bdt_cut: bool,
    efficiency: bool,
    alt_bkg: bool,
):
    """
    Do mass fits in each time bin

    """
    if efficiency:
        assert bdt_cut, "cannot have efficiency with BDT cut (for now)"

    time_bins = TIME_BINS[2:-1]
    mass_bins = definitions.nonuniform_mass_bins((144.0, 147.0), (50, 100, 50))

    # Get the classifier for BDT cut if we need
    bdt_clf = get_clf(year, "dcs", magnetisation) if bdt_cut else None

    # Get the efficiency reweighters if we need
    dcs_weighter = (
        get_reweighter(
            year, "dcs", magnetisation, "both", fit=False, cut=bdt_cut, verbose=True
        )
        if efficiency
        else None
    )
    cf_weighter = (
        get_reweighter(
            year, "cf", magnetisation, "both", fit=False, cut=bdt_cut, verbose=True
        )
        if efficiency
        else None
    )

    plot_dir = mass_util.plot_dir(bdt_cut, efficiency, phsp_bin)

    bkg_str = "alt_bkg_" if alt_bkg else ""

    for i, (low_t, high_t) in enumerate(zip(time_bins[:-1], time_bins[1:])):
        # Get generators of dataframes
        cf_dfs = _dataframes(
            year, magnetisation, phsp_bin, "cf", low_t, high_t, bdt_clf, bdt_cut
        )
        dcs_dfs = _dataframes(
            year, magnetisation, phsp_bin, "dcs", low_t, high_t, bdt_clf, bdt_cut
        )

        # Find efficiency weights if necessary
        # Get new generators to avoid using the old ones up
        cf_wts = (
            wts_generator(
                _dataframes(
                    year, magnetisation, phsp_bin, "cf", low_t, high_t, bdt_clf, bdt_cut
                ),
                cf_weighter,
            )
            if efficiency
            else None
        )
        dcs_wts = (
            wts_generator(
                _dataframes(
                    year,
                    magnetisation,
                    phsp_bin,
                    "dcs",
                    low_t,
                    high_t,
                    bdt_clf,
                    bdt_cut,
                ),
                dcs_weighter,
            )
            if efficiency
            else None
        )

        # Find counts
        cf_count, cf_err = mass_util.delta_m_counts(cf_dfs, mass_bins, cf_wts)
        dcs_count, dcs_err = mass_util.delta_m_counts(dcs_dfs, mass_bins, dcs_wts)

        _fit(
            cf_count,
            dcs_count,
            cf_err,
            dcs_err,
            i,
            mass_bins,
            plot_dir,
            alt_bkg=alt_bkg,
            bdt_cut=bdt_cut,
            efficiency=efficiency,
        )
        _separate_fit(
            cf_count,
            cf_err,
            i,
            mass_bins,
            "cf",
            f"{plot_dir}{bkg_str}rs/{i}.png",
            alt_bkg=alt_bkg,
            bdt_cut=bdt_cut,
            efficiency=efficiency,
        )
        _separate_fit(
            dcs_count,
            dcs_err,
            i,
            mass_bins,
            "dcs",
            f"{plot_dir}{bkg_str}ws/{i}.png",
            alt_bkg=alt_bkg,
            bdt_cut=bdt_cut,
            efficiency=efficiency,
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

    main(**vars(parser.parse_args()))
