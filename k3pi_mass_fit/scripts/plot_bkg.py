"""
Find what the background distribution should look like
by finding the invariant masses M(K3pi) and M(K3pi+pi_s)
where the pi_s comes from a different event

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from libFit import bkg, pdfs


def _plot(
    axis: plt.Axes, bins: np.ndarray, counts: np.ndarray, errs: np.ndarray, **plot_kw
) -> None:
    """
    Plot on an axis

    """
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    scale_factor = np.sum(counts) * widths

    axis.errorbar(
        centres,
        counts / scale_factor,
        xerr=widths,
        yerr=errs / scale_factor,
        fmt=".",
        markersize=0.1,
        **plot_kw,
    )


def main(*, year: str, magnetisation: str, bdt_cut: bool):
    """
    Get the dumps, plot them

    """
    bins = np.linspace(pdfs.domain()[0], 160, 250)

    fig, axis = plt.subplots(1, 1, figsize=(5, 5))

    _plot(
        axis,
        bins,
        *bkg.get_counts(year, magnetisation, "dcs", bins, bdt_cut=bdt_cut),
        color="r",
        label="DCS",
    )
    _plot(
        axis,
        bins,
        *bkg.get_counts(year, magnetisation, "cf", bins, bdt_cut=bdt_cut),
        color="b",
        label="CF",
    )

    axis.legend()

    axis.set_xlabel(r"$\Delta M$")
    axis.set_ylabel("count / MeV")

    upper = pdfs.domain()[1]
    axis.axvline(upper, color="k", alpha=0.8)
    axis.arrow(upper, 0.06, -2, 0.0, color="k", head_length=0.5)
    axis.text(upper - 5, 0.0625, "Fit region", color="k", alpha=0.8)

    fig.tight_layout()

    path = f"bkg_shape{'_cut' if bdt_cut else ''}.png"

    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot pickle dumps of arrays for the empirical bkg shape"
    )
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
        "--bdt_cut",
        action="store_true",
        help="whether to BDT cut the dataframes before finding bkg",
    )

    main(**vars(parser.parse_args()))
