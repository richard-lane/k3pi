"""
For datafames where it is applicable (particle gun, full MC, real data), plot
the D0 and D* masses and their mass difference

For now - just real data

"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, definitions


def _d0_mass(df: pd.DataFrame) -> np.ndarray:
    """D0 Mass"""
    return df["D0 mass"]


def _dst_mass(df: pd.DataFrame) -> np.ndarray:
    """D* Mass"""
    return df["D* mass"]


def _delta_mass(df: pd.DataFrame) -> np.ndarray:
    """D* - D0 Mass"""
    return _dst_mass(df) - _d0_mass(df)


def _plot(axis: np.ndarray, dataframe: pd.DataFrame, label: str) -> None:
    """
    Plot D masses and Delta M on an array of axes

    """
    d0_bins = np.linspace(1830, 1900, 100)
    dst_bins = np.linspace(1960, 2060, 100)
    delta_m_bins = np.linspace(125, 160, 100)

    hist_kw = {"density": True, "histtype": "step"}

    axis[0].hist(_d0_mass(dataframe), bins=d0_bins, **hist_kw, label=label)
    axis[1].hist(_dst_mass(dataframe), bins=dst_bins, **hist_kw, label=label)
    axis[2].hist(_delta_mass(dataframe), bins=delta_m_bins, **hist_kw, label=label)


def main():
    """
    Create plots

    """
    year, magnetisation = "2018", "magdown"

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    if os.path.exists(definitions.data_dir(year, "cf", magnetisation)):
        _plot(
            ax,
            pd.concat(get.data(year, "cf", magnetisation), ignore_index=True),
            "CF Data",
        )

    if os.path.exists(definitions.data_dir(year, "dcs", magnetisation)):
        _plot(
            ax,
            pd.concat(get.data(year, "dcs", magnetisation), ignore_index=True),
            "DCS Data",
        )

    if os.path.exists(definitions.mc_dump(year, "cf", magnetisation)):
        _plot(ax, get.mc(year, "cf", magnetisation), "CF MC")

    if os.path.exists(definitions.mc_dump(year, "dcs", magnetisation)):
        _plot(ax, get.mc(year, "dcs", magnetisation), "DCS MC")

    ax[0].set_title("D0 Mass")
    ax[1].set_title("D* Mass")
    ax[2].set_title(r"$\Delta$ Mass")

    fig.tight_layout()

    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
