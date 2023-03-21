"""
Fit to D0 IPCHI2 histogram in time bin 1 only

This should have negligible secondary contamination and
allows us to fix the shape of the prompt peak in IPCHI2

"""
import sys
import pathlib
import argparse
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
from lib_data import get, ipchi2_fit, stats
from lib_time_fit import definitions


def _fit(log_ipchi2s: np.ndarray, param_guess: dict) -> Minuit:
    """
    Perform an unbinned fit to the

    """
    cost_fcn = UnbinnedNLL(log_ipchi2s, ipchi2_fit.norm_peak)

    fitter = Minuit(cost_fcn, **param_guess)

    fitter.fixed["beta"] = True
    fitter.migrad(ncall=5000)

    fitter.fixed["beta"] = False
    fitter.migrad(ncall=5000)

    return fitter


def _ipchi2s(sign: str) -> np.ndarray:
    """
    ipchi2 in the given time range

    """
    # Make it run faster; set to None for all
    n_dfs = None

    low_t, high_t = definitions.TIME_BINS[1:3]
    print(f"{low_t=}\t{high_t=}")
    dfs = (
        dataframe[(low_t < dataframe["time"]) & (dataframe["time"] < high_t)]
        for dataframe in islice(get.data("2018", sign, "magdown"), n_dfs)
    )
    return np.concatenate([np.log(dataframe["D0 ipchi2"]) for dataframe in dfs])


def main(*, sign: str):
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    ipchi2s = _ipchi2s(sign)

    # Get rid of points outside domain where the PDF is defined
    low_ip, high_ip = ipchi2_fit.domain()
    ipchi2s = ipchi2s[ipchi2s > low_ip]
    ipchi2s = ipchi2s[ipchi2s < high_ip]

    # fit
    fitter = _fit(
        ipchi2s,
        {
            "centre": 1.5,
            "width_l": 2.5,
            "width_r": 2.5,
            "alpha_l": 0.1,
            "alpha_r": 0.1,
            "beta": 0.0,
        },
    )

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

    pts = np.linspace(low_ip, high_ip, 1_000)
    predicted = (
        stats.areas(bins, len(ipchi2s) * ipchi2_fit.norm_peak(bins, *fitter.values))
        / bin_widths
    )
    axes["A"].plot(centres, predicted)

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
        str(dict(zip(fitter.parameters, [f"{float(x):.3f}" for x in fitter.values])))
        + f"\n{fitter.valid=}"
    )

    fig.savefig(f"ipchi2_fit_lowtime_{sign}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform IPCHI2 fit to the low decay times only to fix the shape of"
        "the prompt peak",
    )

    parser.add_argument("sign", type=str, choices={"dcs", "cf"}, help="data type")

    main(**vars(parser.parse_args()))
