"""
Mass fits to data to get the yields

Outputs them to a file

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

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


def _fit(
    rs_count: np.ndarray,
    ws_count: np.ndarray,
    rs_err: np.ndarray,
    ws_err: np.ndarray,
    bin_number: int,
    bins: np.ndarray,
    n_underflow: int,
    fit_dir: str,
) -> Tuple[float, float, float, float, float]:
    """
    Plot the fit

    Returns fval, N RS, N RS err, N WS, N WS err

    """
    rs_total = np.sum(rs_count)
    ws_total = np.sum(ws_count)
    initial_guess = (
        rs_total * 0.9,
        rs_total * 0.1,
        ws_total * 0.05,
        ws_total * 0.95,
        *mass_util.signal_param_guess(bin_number),
        *mass_util.sqrt_bkg_param_guess("cf"),
        *mass_util.sqrt_bkg_param_guess("dcs"),
    )
    try:
        fitter = fit.binned_simultaneous_fit(
            rs_count[n_underflow:],
            ws_count[n_underflow:],
            bins[n_underflow:],
            initial_guess,
            (bins[n_underflow], bins[-1]),
            rs_errors=rs_err[n_underflow:],
            ws_errors=ws_err[n_underflow:],
        )
    except pdfs.ZeroCountsError as err:
        print("time bin:", bin_number, repr(err))
        return

    params = fitter.values
    print(f"{fitter.valid=}", end="\t")
    print(f"{params[-2]} +- {fitter.errors[-2]}", end="\t")
    print(f"{params[-1]} +- {fitter.errors[-1]}")

    fig, axes = plotting.simul_fits(
        rs_count,
        rs_err,
        ws_count,
        ws_err,
        bins,
        (bins[n_underflow], bins[-1]),
        params,
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

    plot_path = f"{fit_dir}yield_fit_{bin_number}.png"

    print(f"Saving {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)

    return (
        fitter.fval,
        fitter.values[0],
        fitter.errors[0],
        fitter.values[2],
        fitter.errors[2],
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
        # Phsp bin
        get.binned_generator(
            # Time bin
            get.time_binned_generator(
                get.data(year, sign, magnetisation), low_t, high_t
            ),
            phsp_bin,
        ),
        bdt_clf,
        perform_cut=bdt_cut,
    )


def _abs_eff(sign: str, time_range: Tuple[float, float]) -> float:
    """
    Get the absolute efficiency, accepting only times in the provided range

    """
    n_gen = get.pgun_n_generated(sign)

    times = get.particle_gun(sign)["time"]
    low, high = time_range
    accepted = (low < times) & (times < high)

    return np.sum(accepted) / n_gen


def main(
    *,
    year: str,
    magnetisation: str,
    phsp_bin: int,
    bdt_cut: bool,
    efficiency: bool,
):
    """
    Do mass fits in each time bin

    """
    time_bins = TIME_BINS[2:-1]

    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], 144.5, 146.5, high), (n_underflow, 50, 100, 50)
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
    plot_dir = mass_util.plot_dir(bdt_cut, efficiency, phsp_bin)

    # For writing the result
    yield_file_path = mass_util.yield_file(
        year, magnetisation, phsp_bin, bdt_cut, efficiency
    )

    # If the file already exists, appending to it might have unexpected results
    yield_file_path.touch(exist_ok=False)

    # Find the absolute efficiency of the events in our time acceptance range
    dcs_eff = _abs_eff("dcs", (time_bins[0], time_bins[1]))
    cf_eff = _abs_eff("cf", (time_bins[0], time_bins[1]))

    for i, (low_t, high_t) in enumerate(zip(time_bins[:-1], time_bins[1:])):
        print("finding cf counts - no efficiency")
        cf_count, cf_err = mass_util.delta_m_counts(
            _dataframes(
                year, magnetisation, phsp_bin, "cf", low_t, high_t, bdt_clf, bdt_cut
            ),
            mass_bins,
        )
        print("finding dcs counts - no efficiency")
        dcs_count, dcs_err = mass_util.delta_m_counts(
            _dataframes(
                year, magnetisation, phsp_bin, "dcs", low_t, high_t, bdt_clf, bdt_cut
            ),
            mass_bins,
        )

        # Find efficiency weights if necessary
        if efficiency:
            cf_wts = wts_generator(
                _dataframes(
                    year, magnetisation, phsp_bin, "cf", low_t, high_t, bdt_clf, bdt_cut
                ),
                cf_weighter,
            )
            dcs_wts = wts_generator(
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

            print("finding cf counts - with efficiency")
            cf_count_eff, cf_err = mass_util.delta_m_counts(
                _dataframes(
                    year, magnetisation, phsp_bin, "cf", low_t, high_t, bdt_clf, bdt_cut
                ),
                mass_bins,
                cf_wts,
            )
            print("finding dcs counts - with efficiency")
            dcs_count_eff, dcs_err = mass_util.delta_m_counts(
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
                mass_bins,
                dcs_wts,
            )

            # scale the dcs counts to account for absolute efficiency
            scale_factor = np.sum(cf_count_eff) / np.sum(dcs_count_eff)
            scale_factor *= np.sum(dcs_count) / np.sum(cf_count)
            scale_factor *= cf_eff / dcs_eff
            print(f"{scale_factor=}")

            dcs_count = dcs_count_eff * scale_factor
            dcs_err = dcs_err * scale_factor

            cf_count = cf_count_eff

        chi2, n_rs, err_rs, n_ws, err_ws = _fit(
            cf_count,
            dcs_count,
            cf_err,
            dcs_err,
            i,
            mass_bins,
            n_underflow,
            plot_dir,
        )

        # Write results to a file
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

    main(**vars(parser.parse_args()))
