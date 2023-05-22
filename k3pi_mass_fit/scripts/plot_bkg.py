"""
Find what the background distribution should look like
by finding the invariant masses M(K3pi) and M(K3pi+pi_s)
where the pi_s comes from a different event

"""
import sys
import pickle
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


def _plot_pdf(
    axis: plt.Axes,
    bins: np.ndarray,
    year: str,
    sign: str,
    magnetisation: str,
    *,
    bdt_cut: bool,
    colour: str,
) -> None:
    """
    Plot the PDF derived from the counts on an axis

    """
    pdf = bkg.pdf(bins, year, magnetisation, sign, bdt_cut=bdt_cut)

    pts = np.linspace(bins[0], bins[-1], 200)
    fval = pdf(pts)

    axis.plot(pts, fval, color=colour, linewidth=0.5)


def _count_in_fit_region(
    year: str, sign: str, magnetisation: str, *, bdt_cut: str
) -> float:
    """
    Find the count in teh fit region by findnig the counts in one big bin

    """
    bins = [-np.inf, *pdfs.reduced_domain(), np.inf]
    counts, _ = bkg.get_counts(year, magnetisation, sign, bins, bdt_cut=bdt_cut)
    return counts[1]


def main(*, year: str, magnetisation: str, bdt_cut: bool):
    """
    Get the dumps, plot them

    """
    bins = np.linspace(pdfs.domain()[0], 160, 250)
    reduced_bins = np.linspace(*pdfs.reduced_domain(), 250)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    _plot(
        axes[0],
        bins,
        *bkg.get_counts(year, magnetisation, "dcs", bins, bdt_cut=bdt_cut),
        color="r",
        label="DCS",
    )
    _plot(
        axes[0],
        bins,
        *bkg.get_counts(year, magnetisation, "cf", bins, bdt_cut=bdt_cut),
        color="b",
        label="CF",
    )

    _plot_pdf(
        axes[1],
        reduced_bins,
        year,
        "dcs",
        magnetisation,
        bdt_cut=bdt_cut,
        colour="r",
    )
    _plot_pdf(
        axes[1],
        reduced_bins,
        year,
        "cf",
        magnetisation,
        bdt_cut=bdt_cut,
        colour="b",
    )

    axes[0].legend()

    for axis in axes:
        axis.set_xlabel(r"$\Delta M$")
        axis.set_ylabel("count / MeV")

        lower, upper = pdfs.reduced_domain()
        axis.axvline(lower, color="k", alpha=0.8)
        axis.axvline(upper, color="k", alpha=0.8)
    axes[0].arrow(upper, 0.06, -2, 0.0, color="k", head_length=0.5)
    axes[0].text(upper - 5, 0.0625, "Fit region", color="k", alpha=0.8)

    fig.tight_layout()

    path = f"bkg_shape{'_cut' if bdt_cut else ''}.png"

    print(f"plotting {path}")
    fig.savefig(path)
    with open(path, "wb") as f:
        pickle.dump((fig, axes), f"plot_pkls/{path}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot pickle dumps of arrays for the empirical bkg shape"
    )
    parser.add_argument(
        "year",
        type=str,
        choices={"2017", "2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )
    parser.add_argument(
        "--bdt_cut",
        action="store_true",
        help="whether to BDT cut the dataframes before finding bkg",
    )

    main(**vars(parser.parse_args()))
