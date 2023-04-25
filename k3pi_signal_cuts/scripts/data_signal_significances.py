"""
At various values of the BDT threshold,
do mass fits to the real data and measure the signal
significance

Then plot

"""
import sys
import pathlib
import argparse
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from lib_cuts import get as get_bdt
from lib_cuts import metrics
from lib_cuts.definitions import THRESHOLD
from libFit import definitions, pdfs
from libFit.fit import binned_simultaneous_fit
from libFit import util
from lib_data import get, stats, cuts
from lib_time_fit.definitions import TIME_BINS


def _dataframes(
    year: str, sign: str, magnetisation: str, clf, threshold: float
) -> Iterable[pd.DataFrame]:
    """
    Get the dataframes for the mass ft

    """
    low_t, high_t = TIME_BINS[2:4]
    phsp_bin = 0
    generator = get.binned_generator(
        get.time_binned_generator(
            cuts.cands_cut_dfs(
                cuts.ipchi2_cut_dfs(get.data(year, sign, magnetisation))
            ),
            low_t,
            high_t,
        ),
        phsp_bin,
    )

    # Do BDT cut
    for dataframe in generator:
        yield get_bdt.signal_cut_df(dataframe, clf, threshold)


def main(*, year: str, magnetisation: str):
    """
    Scan significances in real data

    """
    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], high), (n_underflow, 150)
    )

    # Define threshholds
    n_thresholds = 20
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]

    significances = []
    errors = []
    plot_thresholds = []

    clf = get_bdt.classifier(year, "dcs", magnetisation)

    # Repeat the below fit for each threshold
    for threshold in tqdm(thresholds):
        # Get data
        rs_count, rs_err = stats.counts_generator(
            util.delta_m_generator(
                _dataframes(year, "cf", magnetisation, clf, threshold)
            ),
            mass_bins,
        )
        ws_count, ws_err = stats.counts_generator(
            util.delta_m_generator(
                _dataframes(year, "dcs", magnetisation, clf, threshold)
            ),
            mass_bins,
        )

        # Do mass fit
        rs_total = np.sum(rs_count)
        ws_total = np.sum(ws_count)
        initial_guess = (
            rs_total * 0.9,
            rs_total * 0.1,
            ws_total * 0.05,
            ws_total * 0.95,
            *util.signal_param_guess(2),
            *util.sqrt_bkg_param_guess("cf"),
            *util.sqrt_bkg_param_guess("dcs"),
        )
        try:
            fitter = binned_simultaneous_fit(
                rs_count[n_underflow:],
                ws_count[n_underflow:],
                mass_bins[n_underflow:],
                initial_guess,
                (mass_bins[n_underflow], mass_bins[-1]),
                rs_errors=rs_err[n_underflow:],
                ws_errors=ws_err[n_underflow:],
            )
        except pdfs.ZeroCountsError:
            print(f"0 count {threshold=}")
            break

        # Calculate signal significance
        n_sig, n_bkg = fitter.values[2], fitter.values[3]

        significances.append(metrics.signal_significance(n_sig, n_bkg))
        plot_thresholds.append(threshold)

        # Calculate error on signal significance

    # Plot them
    fig, axis = plt.subplots()
    axis.plot(plot_thresholds, significances, "k.")
    axis.axvline(THRESHOLD, color="r")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Significance")
    fig.tight_layout()
    fig.savefig(f"data_signal_significances_{year}_{magnetisation}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do mass fits in time bin 0, phsp bin 1 to assess how much sig/bkg there is"
    )
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
    main(**vars(parser.parse_args()))
