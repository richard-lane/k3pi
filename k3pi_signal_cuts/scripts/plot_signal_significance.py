"""
Plot signal significance of the testing sample

"""
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts import util, metrics, definitions
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def main(*, year: str, sign: str, magnetisation: str):
    """
    We have to make a choice about the threshhold value for predict_proba above which we consider
    an event to be signal. By default in sklearn this is 0.5, but maybe we will find a better signal
    significance by choosing a different value.

    Note that the values returned by predict_proba may not correspond exactly to probabilities.
    This can be checked by running the calibration curve script, but chances are that it's good
    enough.

    Plots signal significances for various values of this threshhold.

    """
    # Read dataframes of stuff
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # throw away data to get realistic yields
    gen = np.random.default_rng()
    sig_df = util.discard(gen, sig_df, definitions.EXPECTED_N_SIG_SIG_REGION)
    bkg_df = util.discard(gen, bkg_df, definitions.EXPECTED_N_BKG_SIG_REGION)

    print(
        f"sig frac {len(sig_df):,} / {len(sig_df) + len(bkg_df):,}"
        f"= {100 * len(sig_df) / (len(sig_df) + len(bkg_df)):.4f}%"
    )

    # Find signal probabilities
    clf = get_clf(year, sign, magnetisation)
    training_var_names = list(training_vars.training_var_names())
    sig_probs = clf.predict_proba(sig_df[training_var_names])[:, 1]
    bkg_probs = clf.predict_proba(bkg_df[training_var_names])[:, 1]

    # For various values of the threshhold, find the signal significance
    def sig(threshhold: float) -> float:
        n_sig = np.sum(sig_probs > threshhold)
        n_bkg = np.sum(bkg_probs > threshhold)

        # Scale the number of signal and background by the expected amount of signal
        total_expected = 1697700  # Total DCS magdown expected in signal region
        scale_factor = total_expected / (len(sig_probs) + len(bkg_probs))

        return metrics.signal_significance(scale_factor * n_sig, scale_factor * n_bkg)

    x_range = 0.0, 0.90
    threshholds = np.linspace(*x_range, 51)
    significances = [sig(threshhold) for threshhold in threshholds]

    # Find the max of values
    max_index = np.nanargmax(significances)
    max_response = significances[max_index]
    max_threshhold = threshholds[max_index]

    # Interpolate the threshholds
    lots_of_points = np.linspace(*x_range, 1000)
    response = interp1d(threshholds, significances)(lots_of_points)

    fig, ax = plt.subplots()
    ax.plot(threshholds, significances, "k+")
    ax.plot(lots_of_points, response, "k--")
    ax.plot(max_threshhold, max_response, "ro")

    # Plot an arrow
    length = 100
    plt.arrow(
        max_threshhold,
        max_response - length,
        0,
        length,
        length_includes_head=True,
        color="r",
    )
    plt.text(max_threshhold, max_response - 1.1 * length, f"{max_threshhold=:.3f}")

    ax.set_xlabel("probability threshhold")
    ax.set_ylabel("signal significance")

    plt.savefig(f"significance_threshholds_{year}_{sign}_{magnetisation}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot signal significance for the expected stats"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(**vars(parser.parse_args()))
