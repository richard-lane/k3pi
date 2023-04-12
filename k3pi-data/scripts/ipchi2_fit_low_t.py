"""
Fit to D0 IPCHI2 histogram in time bin 1 only

This should have negligible secondary contamination and
allows us to fix the shape of the prompt peak in IPCHI2

"""
import sys
import pickle
import pathlib
import argparse
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
from lib_data import get, ipchi2_fit, stats, cuts
from lib_time_fit import definitions


def _fit(log_ipchi2s: np.ndarray, param_guess: dict) -> Minuit:
    """
    Perform an unbinned fit to the

    """
    cost_fcn = UnbinnedNLL(log_ipchi2s, ipchi2_fit.norm_peak)

    fitter = Minuit(cost_fcn, **param_guess)

    fitter.limits = (
        (0.8, 2.5),  # Centre
        (0.8, 2.5),  # width L
        (0.8, 2.5),  # width R
        (0.0, 2.0),  # alpha L
        (0.0, 2.0),  # alpha R
        (None, None),  # Beta (fixed below)
    )

    fitter.fixed["beta"] = True
    fitter.migrad(ncall=5000)

    fitter.fixed["beta"] = False
    fitter.migrad(ncall=5000)

    return fitter


def _ipchi2s(year: str, sign: str, magnetisation: str) -> np.ndarray:
    """
    ipchi2 in the given time range

    """
    # Make it run faster; set to None for all
    n_dfs = None

    low_t, high_t = definitions.TIME_BINS[1:3]
    print(f"{low_t=}\t{high_t=}")
    dfs = (
        dataframe[(low_t < dataframe["time"]) & (dataframe["time"] < high_t)]
        for dataframe in islice(get.data(year, sign, magnetisation), n_dfs)
    )
    return np.concatenate([np.log(dataframe["D0 ipchi2"]) for dataframe in dfs])


def main(*, year: str, sign: str, magnetisation: str):
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    ipchi2s = _ipchi2s(year, sign, magnetisation)

    # Get rid of points outside domain where the PDF is defined
    low_ip, high_ip = ipchi2_fit.domain()
    ipchi2s = ipchi2s[ipchi2s > low_ip]
    ipchi2s = ipchi2s[ipchi2s < high_ip]

    # Only use first N pts
    n_pts = 1_000_000
    ipchi2s = ipchi2s[:n_pts]

    # fit
    fitter = _fit(
        ipchi2s,
        {
            "centre": 1.45,
            "width_l": 1.0,
            "width_r": 1.0,
            "alpha_l": 0.0,
            "alpha_r": 0.0,
            "beta": 0.0,
        },
    )

    # Dump the fit params to file
    pkl_file = str(ipchi2_fit.sec_frac_lowtime_file(sign))
    with open(pkl_file, "wb") as dump_f:
        pickle.dump(np.array(fitter.values), dump_f)

    # Read them again from file
    fit_params = ipchi2_fit.low_t_params(sign)

    # plot the fit
    fig, axes = plt.subplot_mosaic("AAA\n" * 3 + "BBB", figsize=(15, 15))

    bins = np.linspace(low_ip, high_ip, 250)
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    counts, errs = stats.counts(ipchi2s, bins)
    axes["A"].errorbar(
        centres,
        counts / bin_widths,
        xerr=bin_widths / 2,
        yerr=errs / bin_widths,
        fmt="k+",
    )

    predicted = (
        stats.areas(bins, len(ipchi2s) * ipchi2_fit.norm_peak(bins, *fit_params))
        / bin_widths
    )
    axes["A"].plot(centres, predicted)
    axes["A"].axvline(np.log(cuts.MAX_IPCHI2))

    diff = ((counts / bin_widths) - predicted) / (errs / bin_widths)
    axes["B"].axhline(0, color="k")
    for pos in (-1, 1):
        axes["B"].axhline(pos, color="r", alpha=0.5)

    # Only plot pull for the fit values
    axes["B"].errorbar(
        centres, diff, yerr=1.0, fmt="k.", elinewidth=0.5, markersize=1.0
    )

    axes["B"].set_xlabel(r"IP$\chi^2$")
    axes["A"].set_ylabel("Count / MeV")

    fig.tight_layout()

    print(fitter)
    fig.suptitle(
        str(dict(zip(fitter.parameters, [f"{float(x):.3f}" for x in fit_params])))
        + f"\n{fitter.valid=}"
    )

    fig.savefig(f"ipchi2_fit_lowtime_{sign}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform IPCHI2 fit to the low decay times only to fix the shape of"
        "the prompt peak",
    )

    parser.add_argument(
        "year", type=str, choices={"2016", "2017", "2018"}, help="data taking year"
    )
    parser.add_argument("sign", type=str, choices={"dcs", "cf"}, help="data type")
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
