"""
Example of sWeighting

"""
import sys
import pathlib
import argparse
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_data import get, stats
from lib_cuts.get import cut_dfs, classifier as get_clf
from libFit import fit, pdfs, definitions, plotting, sweighting, util as mass_util
from lib_efficiency.efficiency_definitions import MIN_TIME


def _dfs(
    year: str, magnetisation: str, sign: str, bdt_cut: bool
) -> Iterable[pd.DataFrame]:
    """
    Get a generator of the right dataframes

    """
    dfs = get.data(year, sign, magnetisation)

    # Get the BDT if we need to
    bdt_clf = get_clf(year, "dcs", magnetisation) if bdt_cut else None

    # This fcn doesn't actually modify the dataframes if we
    # don't provide a classifier
    dataframes = cut_dfs(dfs, bdt_clf, perform_cut=bdt_cut)

    # Only return dataframes over a subset of times, for now
    for dataframe in dataframes:
        times = dataframe["time"]
        keep = (MIN_TIME < times) & (times < 10.0)

        yield dataframe[keep]


def _d0_momenta(
    year: str, magnetisation: str, sign: str, bdt_cut: bool
) -> Iterable[np.ndarray]:
    """
    Generator of D0 momenta

    """
    for dataframe in _dfs(year, magnetisation, sign, bdt_cut):
        yield dataframe["D0 P"]


def _massfit(
    year: str,
    magnetisation: str,
    sign: str,
    bdt_cut: bool,
    fit_range: Tuple[float, float],
) -> Tuple:
    """
    Perform a massfit to some dataframes

    Also makes a plot of the mass fit

    """
    low, high = pdfs.domain()

    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], 144.5, 146.5, high), (n_underflow, 50, 100, 50)
    )
    counts, errs = mass_util.delta_m_counts(
        _dfs(year, magnetisation, sign, bdt_cut), mass_bins
    )

    sig_frac_guess = 0.9 if sign == "cf" else 0.05
    total = np.sum(counts)
    initial_guess = (
        sig_frac_guess * total,
        (1 - sig_frac_guess) * total,
        *mass_util.signal_param_guess(),
        *mass_util.sqrt_bkg_param_guess(sign),
    )
    values = fit.binned_fit(
        counts[n_underflow:],
        mass_bins[n_underflow:],
        initial_guess,
        fit_range,
        errors=errs[n_underflow:],
    ).values

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    plotting.mass_fit(axes, counts, errs, mass_bins, fit_range, values)

    path = f"sweight_example_massfit_{year}_{magnetisation}_{sign}_{bdt_cut=}.png"
    print(f"saving {path}")
    fig.savefig(path)
    plt.close(fig)

    return values


def main(*, year: str, magnetisation: str, sign: str, bdt_cut: bool):
    """
    Do a mass fit; use the parameters from this fit
    to find sWeights; plot the sWeighted D0 P distribtion

    """
    fit_range = pdfs.reduced_domain()

    # Do a mass fit to get the fit parameters
    fit_params = _massfit(year, magnetisation, sign, bdt_cut, fit_range)

    # Get a generator of sWeights given these fit parameters
    sweights = sweighting.sweights(
        _dfs(year, magnetisation, sign, bdt_cut), fit_params, fit_range
    )

    # Get D0 momenta
    d0_p_bins = np.linspace(0.0, 600_000, 250)
    d0_count, d0_err = stats.counts_generator(
        _d0_momenta(year, magnetisation, sign, bdt_cut), d0_p_bins
    )
    d0_count_weighted, d0_err_weighted = stats.counts_generator(
        _d0_momenta(year, magnetisation, sign, bdt_cut), d0_p_bins, weights=sweights
    )

    # Plot the weighted D0 momenta
    centres = (d0_p_bins[1:] + d0_p_bins[:-1]) / 2
    widths = (d0_p_bins[1:] - d0_p_bins[:-1]) / 2
    plt.errorbar(centres, d0_count, xerr=widths, yerr=d0_err, fmt="k.", label="Raw")
    plt.errorbar(
        centres,
        d0_count_weighted,
        xerr=widths,
        yerr=d0_err_weighted,
        fmt="r.",
        label="sWeighted",
    )
    plt.show()


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
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="Event type",
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")

    main(**vars(parser.parse_args()))
