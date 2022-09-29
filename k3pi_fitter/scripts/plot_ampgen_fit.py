"""
Plot the ratio of AmpGen decay times and do a fit

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import get
from lib_time_fit import util
from lib_time_fit import definitions
from lib_time_fit import fitter


def _plot(
    axis: plt.Axes, dcs_df: pd.DataFrame, cf_df: pd.DataFrame, label: str
) -> None:
    """
    Plot ratio of decay times on an axis

    """
    ratio, err, centres, widths = util.bin_times(
        dcs_df["time"], cf_df["time"], bins=np.array(definitions.TIME_BINS)
    )

    axis.errorbar(centres, ratio, xerr=widths, yerr=err, fmt="+", label=label)


def _counts(sign: str) -> np.ndarray:
    """Number of times in each bin"""
    return util.bin_times(get.ampgen(sign)["time"])


def main():
    """
    Get dataframes, plot ratios of CF/DCS on an axis

    """
    dcs_t, dcs_err = _counts("dcs")
    cf_t, cf_err = _counts("cf")

    # There will be no points in the first time bin, so just throw these values away
    dcs_t = dcs_t[1:]
    cf_t = cf_t[1:]
    dcs_err = dcs_err[1:]
    cf_err = cf_err[1:]

    ratio, err = util.ratio_err(dcs_t, cf_t, dcs_err, cf_err)

    # We binned with the default bins in _counts, which are
    # definitions.TIME_BINS
    # We also threw away the first bin so only take [1:]
    # Kind of ugly but it's ok
    bins = definitions.TIME_BINS[1:]
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    # No mixing "fit"
    best_rd, err_on_rd = fitter.no_mixing(ratio, err)

    fig, axis = plt.subplots()
    axis.errorbar(centres, ratio, xerr=widths, yerr=err, label="AmpGen")

    axis.set_xlabel(r"t / $\tau$")
    axis.set_ylabel(r"$WS/RS$")

    ratio_str, err_str = f"{best_rd:.4f}", f"{err_on_rd:.4f}"
    best_ratio = best_rd**2
    axis.plot(
        axis.get_xlim(),
        [best_ratio, best_ratio],
        "k--",
        label=r"fit $r_D=" + ratio_str + "\pm" + err_str + "$",
    )

    fig.suptitle("AmpGen Time Ratios")

    axis.legend()

    plt.show()


if __name__ == "__main__":
    main()
