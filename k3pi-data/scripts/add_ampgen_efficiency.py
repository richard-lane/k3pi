"""
Add a column to the AmpGen dataframes for whether each event
was detected or not under some mock efficiency function

"""
import sys
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_efficiency import mock_efficiency
from lib_data import get, definitions


def _plot_kept(axes: np.ndarray, dataframe: pd.DataFrame, sign: str) -> None:
    """
    Plot and show time efficiency

    """
    bins = np.linspace(0, 10, 100)
    hist_kw = {"histtype": "step", "bins": bins}
    kept_count, _, _ = axes[0].hist(
        dataframe["time"][dataframe["accepted"]],
        **hist_kw,
        label=f"{sign} Accepted",
    )
    total_count, _, _ = axes[0].hist(
        dataframe["time"], **hist_kw, label=f"{sign} Total"
    )

    centres = (bins[1:] + bins[:-1]) / 2

    axes[1].plot(centres, kept_count / total_count, label=f"{sign} efficiency")

    for axis in axes:
        axis.legend()
        axis.set_xlabel(r"time /$\tau$")


def _efficiency_column_exists(dataframe: pd.DataFrame) -> bool:
    """whether the efficiency column is already there"""
    return "accepted" in dataframe


def _add_col(rng: np.random.Generator, sign: str, axes: plt.Axes):
    """
    Use rejection sampling to determine whether each event was accepted
    under some mock efficiency function

    Then add columns to the dataframe for whether each event was accepted/rejected

    Finally write the dataframe to disk

    """
    # Read AmpGen dataframe
    dataframe = get.ampgen(sign)

    if _efficiency_column_exists(dataframe):
        print(f"overwriting {sign} column")

    # Use rejection sampling to see which events are kept
    time_factor = 1.0 if sign == "dcs" else 0.99
    phsp_factor = 1.0 if sign == "dcs" else 0.95
    kept = mock_efficiency.accepted(
        rng, dataframe, time_factor=time_factor, phsp_factor=phsp_factor
    )
    print(f"{np.sum(kept)=}\t{len(kept)}")

    # Add a column to the dataframe showing this
    dataframe["accepted"] = kept

    # Plot stuff
    _plot_kept(axes, dataframe, sign)

    # Write to disk
    with open(definitions.ampgen_dump(sign), "wb") as f:
        pickle.dump(dataframe, f)


def main():
    """
    Add columns to DCS and CF dataframes - make plots

    """
    rng = np.random.default_rng(seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    _add_col(rng, "dcs", axes)
    _add_col(rng, "cf", axes)

    fig.tight_layout()
    path = "mock_ampgen_efficiency.png"
    print(f"saving {path}")
    fig.savefig(path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add efficiency and test/train columns to AmpGen dataframe. Also plot stuff"
    )
    parser.parse_args()

    main()
