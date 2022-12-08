"""
Find what the background distribution should look like
by finding the invariant masses M(K3pi) and M(K3pi+pi_s)
where the pi_s comes from a different event

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_cuts.get import cut_dfs, classifier as get_clf
from lib_efficiency.efficiency_util import k_3pi
from lib_data.stats import counts_generator
from lib_data import get, definitions, util


def _invmass_gen(
    rng: np.random.Generator, year: str, sign: str, magnetisation: str, *, bdt_cut: bool
):
    """
    Get a generator of (K3pi - K4pi) invariant masses

    """
    df_generator = get.data(year, sign, magnetisation)
    if bdt_cut:
        clf = get_clf(year, "dcs", magnetisation)
        df_generator = cut_dfs(df_generator, clf)

    for dataframe in df_generator:
        k3pi = k_3pi(dataframe)
        slowpi = np.row_stack(
            [dataframe[f"slowpi_{s}"] for s in definitions.MOMENTUM_SUFFICES]
        )

        # Shift the slow pi
        # Could use np.roll(slowpi, 1, axis=1) to just shift the array by 1
        # Or randomise fully with shuffle
        rng.shuffle(slowpi, axis=1)

        d_mass = util.inv_mass(*k3pi)
        dst_mass = util.inv_mass(*k3pi, slowpi)

        yield dst_mass - d_mass


def _scale(
    bins: np.ndarray, counts: np.ndarray, errs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale counts/errs by the bin widths and the total number

    """
    bin_widths = (bins[1:] - bins[:-1]) / 2

    # Scale by the total number
    total = np.sum(counts)
    errs = errs / total
    counts = counts / total

    # Scale also by the bin widths
    errs /= bin_widths
    counts /= bin_widths

    return counts, errs


def _plot(
    axis: plt.Axes, bins: np.ndarray, counts: np.ndarray, errs: np.ndarray, **plot_kw
) -> None:
    """
    Plot
    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    axis.errorbar(
        centres, counts, xerr=widths, yerr=errs, fmt=".", markersize=0.1, **plot_kw
    )


def _count_err(
    rng: np.random.Generator,
    year: str,
    magnetisation: str,
    bins: np.ndarray,
    *,
    bdt_cut: bool,
) -> Tuple:
    """
    RS and WS counts and errors

    """
    cf_counts, cf_errs = counts_generator(
        _invmass_gen(rng, year, "cf", magnetisation, bdt_cut=bdt_cut), bins
    )
    dcs_counts, dcs_errs = counts_generator(
        _invmass_gen(rng, year, "dcs", magnetisation, bdt_cut=bdt_cut), bins
    )

    # Get rid of the overflow
    bins = bins[:-1]
    cf_counts = cf_counts[:-1]
    cf_errs = cf_errs[:-1]

    dcs_counts = dcs_counts[:-1]
    dcs_errs = dcs_errs[:-1]

    # Scale
    cf_counts, cf_errs = _scale(bins, cf_counts, cf_errs)
    dcs_counts, dcs_errs = _scale(bins, dcs_counts, dcs_errs)

    return cf_counts, cf_errs, dcs_counts, dcs_errs


def main():
    """
    Get the invariant masses from the right dataframes,
    store them in a histogram and plot it

    """
    # Choose dataframes
    year, magnetisation = "2018", "magdown"

    # Choose bins
    bins = np.linspace(139, 152, 200)
    bins = np.concatenate((bins, [np.inf]))

    # Get generator of delta M and bin them
    fig, axes = plt.subplots(1, 2)
    rng = np.random.default_rng(seed=18)

    cf_counts, cf_errs, dcs_counts, dcs_errs = _count_err(
        rng, year, magnetisation, bins, bdt_cut=False
    )
    _plot(axes[0], bins[:-1], dcs_counts, dcs_errs, label="WS")
    _plot(axes[0], bins[:-1], cf_counts, cf_errs, label="RS")
    axes[0].set_title("No BDT cut")
    axes[0].legend()

    cf_counts, cf_errs, dcs_counts, dcs_errs = _count_err(
        rng, year, magnetisation, bins, bdt_cut=True
    )
    _plot(axes[1], bins[:-1], dcs_counts, dcs_errs, label="WS")
    _plot(axes[1], bins[:-1], cf_counts, cf_errs, label="RS")
    axes[1].set_title("With BDT cut")

    for axis in axes:
        axis.set_xlabel(r"M(K3$\pi\pi_s$) - M(K3$\pi$)")

    fig.tight_layout()

    fig.savefig("bkg.png")
    plt.show()


if __name__ == "__main__":
    main()
