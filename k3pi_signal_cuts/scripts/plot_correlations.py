"""
Plot correlation matrices for our BDT variables

Both for simulated signal/bkg and for real data

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_data import get, training_vars


def _add_cbar(fig: plt.Figure, axis: plt.Axes) -> None:
    """
    Add colour bar to figure, using the provided axis as scale

    """
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.30, 0.05, 0.50])

    fig.colorbar(mappable=axis.get_images()[0], cax=cbar_ax)


def _plot(axis: plt.Axes, dataframe: pd.DataFrame) -> None:
    """
    Plot BDT var correlation matrix from a dataframe on an axis

    """
    names = training_vars.training_var_names()
    num = len(names)
    corr = np.ones((num, num))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            corr[i, j] = np.corrcoef(dataframe[name_i], dataframe[name_j])[0, 1]

    axis.imshow(corr, vmin=-1.0, vmax=1.0)

    axis.set_xticks(range(num))
    axis.set_yticks(range(num))

    axis.set_xticklabels(names, rotation=90)
    axis.set_yticklabels(names)


def _plot_simulated():
    """
    Correlation matrices for MC signal / upper mass sideband bkg

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.set_cmap("seismic")

    _plot(ax[0], sig_df)
    _plot(ax[1], bkg_df)

    fig.suptitle("BDT Training var correlation: simulation")

    ax[0].set_title("Signal (MC)")
    ax[1].set_title("Background (data upper mass sideband)")

    fig.tight_layout()

    _add_cbar(fig, ax[0])

    fig.savefig("simulation_correlation.png")

    plt.show()


def _plot_data():
    """
    Plot correlation for real data

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"
    n_dfs = 50
    dataframes = list(
        tqdm(islice(get.data(year, sign, magnetisation), n_dfs), total=n_dfs)
    )
    dataframe = pd.concat(dataframes)

    fig, ax = plt.subplots(figsize=(8, 8))

    _plot(ax, dataframe)

    fig.suptitle("BDT Training var correlation: data")

    fig.tight_layout()
    _add_cbar(fig, ax)

    fig.savefig("data_correlation.png")

    plt.show()


def main():
    """
    Plot correlation matrices

    """
    _plot_simulated()
    _plot_data()


if __name__ == "__main__":
    main()
