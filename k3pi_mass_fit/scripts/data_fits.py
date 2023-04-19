"""
Simultaneous and individual fits to RS and WS

"""
import sys
import pathlib
import argparse
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))

from lib_data import get, cuts
from lib_cuts.get import classifier as get_clf, cut_dfs
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_util import wts_generator
from libFit import pdfs, fit, util as mass_util, plotting, definitions, bkg
from lib_time_fit.definitions import TIME_BINS


def _separate_fit(
    count: np.ndarray,
    err: np.ndarray,
    bin_number: int,
    bins: np.ndarray,
    n_underflow: int,
    sign: str,
    plot_path: str,
    alt_bkg: bool,
    *,
    bkg_pdf: Callable,
):
    """
    Plot separate fits

    """
    sig_frac = 0.9 if sign == "cf" else 0.05
    total = np.sum(count)
    initial_guess = (
        total * sig_frac,
        total * (1 - sig_frac),
        *mass_util.signal_param_guess(bin_number),
        *((0, 0, 0) if alt_bkg else mass_util.sqrt_bkg_param_guess(sign)),
    )
    try:
        fitter = (
            fit.alt_bkg_fit(
                count[n_underflow:],
                bins[n_underflow:],
                initial_guess,
                bkg_pdf,
                errors=err[n_underflow:],
            )
            if alt_bkg
            else fit.binned_fit(
                count[n_underflow:],
                bins[n_underflow:],
                initial_guess,
                (bins[n_underflow], bins[-1]),
                errors=err[n_underflow:],
            )
        )

    except pdfs.ZeroCountsError as exc:
        print("time bin:", bin_number, repr(exc))
        raise exc

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(6, 8))

    if alt_bkg:
        plotting.alt_bkg_fit(
            (axes["A"], axes["B"]),
            count,
            err,
            bins,
            bkg_pdf,
            (bins[n_underflow], bins[-1]),
            fitter.values,
        )
    else:
        plotting.mass_fit(
            (axes["A"], axes["B"]),
            count,
            err,
            bins,
            (bins[n_underflow], bins[-1]),
            fitter.values,
        )

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
    n_underflow: int,
    fit_dir: str,
    alt_bkg: bool,
    *,
    rs_bkg_pdf: Callable,
    ws_bkg_pdf: Callable,
) -> None:
    """
    Plot the fit

    """
    rs_total = np.sum(rs_count)
    ws_total = np.sum(ws_count)
    initial_guess = (
        rs_total * 0.9,
        rs_total * 0.1,
        ws_total * 0.05,
        ws_total * 0.95,
        *mass_util.signal_param_guess(bin_number),
        *((0, 0, 0) if alt_bkg else mass_util.sqrt_bkg_param_guess("cf")),
        *((0, 0, 0) if alt_bkg else mass_util.sqrt_bkg_param_guess("dcs")),
    )
    try:
        fitter = (
            fit.alt_simultaneous_fit(
                rs_count[n_underflow:],
                ws_count[n_underflow:],
                bins[n_underflow:],
                initial_guess,
                rs_bkg_pdf,
                ws_bkg_pdf,
                rs_errors=rs_err[n_underflow:],
                ws_errors=ws_err[n_underflow:],
            )
            if alt_bkg
            else fit.binned_simultaneous_fit(
                rs_count[n_underflow:],
                ws_count[n_underflow:],
                bins[n_underflow:],
                initial_guess,
                (bins[n_underflow], bins[-1]),
                rs_errors=rs_err[n_underflow:],
                ws_errors=ws_err[n_underflow:],
            )
        )
    except pdfs.ZeroCountsError as err:
        print("time bin:", bin_number, repr(err))
        raise err

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
            (bins[n_underflow], bins[-1]),
            rs_bkg_pdf,
            ws_bkg_pdf,
            params,
        )
        if alt_bkg
        else plotting.simul_fits(
            rs_count,
            rs_err,
            ws_count,
            ws_err,
            bins,
            (bins[n_underflow], bins[-1]),
            params,
        )
    )
    n_rs_sig = fitter.values[0]
    n_rs_bkg = fitter.values[1]
    n_ws_sig = fitter.values[2]
    n_ws_bkg = fitter.values[3]

    n_rs_sig_err = fitter.errors[0]
    n_rs_bkg_err = fitter.errors[1]
    n_ws_sig_err = fitter.errors[2]
    n_ws_bkg_err = fitter.errors[3]

    axes["A"].set_title(
        f"Nsig={n_rs_sig:,.2f}"
        r"$\pm$"
        f"{n_rs_sig_err:,.2f}"
        f"\tNbkg={n_rs_bkg:,.2f}"
        r"$\pm$"
        f"{n_rs_bkg_err:,.2f}"
    )
    axes["B"].set_title(
        f"Nsig={n_ws_sig:,.2f}"
        r"$\pm$"
        f"{n_ws_sig_err:,.2f}"
        f"\tNbkg={n_ws_bkg:,.2f}"
        r"$\pm$"
        f"{n_ws_bkg_err:,.2f}"
    )
    fig.suptitle(f"{fitter.valid=}")
    fig.tight_layout()
    fig.tight_layout()

    plot_path = f"{fit_dir}fit_{bin_number}.png"

    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)

    return (
        n_rs_sig,
        n_rs_sig_err,
        n_ws_sig,
        n_ws_sig_err,
    )


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
                cuts.cands_cut_dfs(
                    cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation))
                ),
                low_t,
                high_t,
            ),
            phsp_bin,
        ),
        bdt_clf,
        perform_cut=bdt_cut,
    )


def _n_total(year: str, magnetisation: str, sign: str, bdt_clf) -> int:
    """
    Total length of a generator of dataframes

    All phase space bins (not Ks veto), all times

    """
    n_tot = 0
    for dataframe in _dataframes(
        year,
        magnetisation,
        phsp_bin=None,  # All phsp bins - should we really include the Ks veto bin too?
        sign=sign,
        low_t=0,
        high_t=np.inf,
        bdt_clf=bdt_clf,
        bdt_cut=True,
    ):
        n_tot += len(dataframe)

    return n_tot


def _sum_eff_wt(year: str, magnetisation: str, sign: str, bdt_clf, eff_weighter) -> int:
    """
    Total sum off efficiency weights

    All phase space bins (not Ks veto), all times

    """
    sum_tot = 0
    for wts in wts_generator(
        _dataframes(
            year,
            magnetisation,
            phsp_bin=None,  # All phsp bins - should we really include the Ks veto bin too?
            sign=sign,
            low_t=0,
            high_t=np.inf,
            bdt_clf=bdt_clf,
            bdt_cut=True,
        ),
        eff_weighter,
    ):
        sum_tot += np.sum(wts)

    return sum_tot


def _dcs_eff_wt_scale(
    year: str, magnetisation: str, dcs_weighter, cf_weighter, bdt_clf
) -> float:
    """
    Scaling factor to apply to DCS weights such that we have the right efficiency

    """
    # Get total length of dfs (after BDT cut)
    n_ws = _n_total(year, magnetisation, "dcs", bdt_clf)
    print(f"{n_ws=}", end="\t")
    n_rs = _n_total(year, magnetisation, "cf", bdt_clf)
    print(f"{n_rs=}", end="\t")

    # Get total sum of efficiency weights
    sum_ws = _sum_eff_wt(year, magnetisation, "dcs", bdt_clf, dcs_weighter)
    print(f"{sum_ws=}", end="\t")
    sum_rs = _sum_eff_wt(year, magnetisation, "cf", bdt_clf, cf_weighter)
    print(f"{sum_rs=}", end="\t")

    # Get the average efficiencies from particle gun
    ws_abs_eff = get.absolute_efficiency(year, "dcs", magnetisation)
    print(f"{ws_abs_eff=}", end="\t")
    rs_abs_eff = get.absolute_efficiency(year, "cf", magnetisation)
    print(f"{rs_abs_eff=}", end="\t")

    # Find the right scaling factor to apply to ws weights
    scale_factor = (n_ws / n_rs) * (rs_abs_eff / ws_abs_eff) * (sum_rs / sum_ws)
    print(f"{scale_factor=}")

    return scale_factor


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

    # For writing the result
    yield_file_path = mass_util.yield_file(
        year, magnetisation, phsp_bin, bdt_cut, efficiency, alt_bkg
    )

    # If the file already exists, appending to it might have unexpected results
    yield_file_path.touch(exist_ok=False)

    time_bins = TIME_BINS[2:-1]

    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], high), (n_underflow, 150)
    )

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

    # Get the efficiency weight scaling factor
    if efficiency:
        dcs_eff_wt_scale = _dcs_eff_wt_scale(
            year, magnetisation, dcs_weighter, cf_weighter, bdt_clf
        )

    plot_dir = mass_util.plot_dir(bdt_cut, efficiency, phsp_bin, alt_bkg)

    # Get the bkg pdfs if we need
    if alt_bkg:
        rs_bkg_pdf = bkg.pdf(mass_bins, year, magnetisation, "cf", bdt_cut=bdt_cut)
        ws_bkg_pdf = bkg.pdf(mass_bins, year, magnetisation, "dcs", bdt_cut=bdt_cut)

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
            (
                dcs_eff_wt_scale * wts
                for wts in wts_generator(
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
            )
            if efficiency
            else None
        )

        # Find counts
        cf_count, cf_err = mass_util.delta_m_counts(cf_dfs, mass_bins, cf_wts)
        dcs_count, dcs_err = mass_util.delta_m_counts(dcs_dfs, mass_bins, dcs_wts)

        # Return the yield from the simultaneous fit
        try:
            n_rs, err_rs, n_ws, err_ws = _fit(
                cf_count,
                dcs_count,
                cf_err,
                dcs_err,
                i,
                mass_bins,
                n_underflow,
                plot_dir,
                alt_bkg,
                rs_bkg_pdf=rs_bkg_pdf if alt_bkg else None,
                ws_bkg_pdf=ws_bkg_pdf if alt_bkg else None,
            )
        except pdfs.ZeroCountsError:
            print(f"Simultaneous fit: {i} ZeroCountsError")

        # Plot the individual fits just to check stuff
        try:
            _separate_fit(
                cf_count,
                cf_err,
                i,
                mass_bins,
                n_underflow,
                "cf",
                f"{plot_dir}rs/{i}.png",
                alt_bkg,
                bkg_pdf=rs_bkg_pdf if alt_bkg else None,
            )
        except pdfs.ZeroCountsError:
            print(f"RS: {i} ZeroCountsError")
            continue

        try:
            _separate_fit(
                dcs_count,
                dcs_err,
                i,
                mass_bins,
                n_underflow,
                "dcs",
                f"{plot_dir}ws/{i}.png",
                alt_bkg,
                bkg_pdf=ws_bkg_pdf if alt_bkg else None,
            )
        except pdfs.ZeroCountsError:
            print(f"WS: {i} ZeroCountsError")
            continue

        # Append the yields to the yield file
        mass_util.write_yield(
            (low_t, high_t),
            (n_rs, n_ws),
            (err_rs, err_ws),
            yield_file_path,
            print_str=True,
        )


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
    parser.add_argument(
        "phsp_bin",
        type=int,
        choices=range(4),
        help="Phase space bin index. If not provided, performs phase space integrated fits",
        nargs="?",
        default=None,
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

    main(**vars(parser.parse_args()))
