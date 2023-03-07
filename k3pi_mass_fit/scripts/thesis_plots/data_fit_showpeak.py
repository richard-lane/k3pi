"""
Individual fit to WS showing the peak

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi_fitter"))

from lib_data import get, stats
from lib_cuts.get import classifier as get_clf, cut_dfs
from libFit import pdfs, fit, util as mass_util, definitions


def _separate_fit(
    count: np.ndarray,
    err: np.ndarray,
    bins: np.ndarray,
    n_underflow: int,
    plot_path: str,
    nocut_count: np.ndarray = None,
):
    """
    Plot separate fits

    """
    sig_frac = 0.05
    total = np.sum(count)
    initial_guess = (
        total * sig_frac,
        total * (1 - sig_frac),
        *mass_util.signal_param_guess(5),
        *mass_util.sqrt_bkg_param_guess("dcs"),
    )
    fitter = fit.binned_fit(
        count[n_underflow:],
        bins[n_underflow:],
        initial_guess,
        (bins[n_underflow], bins[-1]),
        errors=err[n_underflow:],
    )
    fit_params = fitter.values

    fig, axis = plt.subplots(figsize=(8, 8))

    # Plot hist
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = (bins[1:] + bins[:-1]) / 2
    axis.hist(
        bins[n_underflow:][:-1],
        bins[n_underflow:],
        weights=count[n_underflow:],
        histtype="step",
        color="k" if nocut_count is None else "darkslategrey",
    )

    # Plot bkg
    predicted_bkg = fit_params[1] * stats.areas(
        bins[n_underflow:],
        pdfs.normalised_bkg(
            bins[n_underflow:], *fit_params[-2:], pdfs.reduced_domain()
        ),
    )
    axis.plot(centres[n_underflow:], predicted_bkg, "r-")

    # Plot signal
    predicted = stats.areas(
        bins[n_underflow:],
        pdfs.model(bins[n_underflow:], *fit_params, pdfs.reduced_domain()),
    )
    axis.plot(centres[n_underflow:], predicted, "indigo")

    predicted_sig = fit_params[0] * stats.areas(
        bins[n_underflow:],
        pdfs.normalised_signal(
            bins[n_underflow:], *fit_params[2:-2], pdfs.reduced_domain()
        ),
    )
    axis.fill_between(
        centres[n_underflow:],
        np.zeros_like(predicted_sig),
        predicted_sig,
        facecolor="green",
    )

    # If nocut count specified, plot that too
    if nocut_count is not None:
        axis.hist(
            bins[n_underflow:][:-1],
            bins[n_underflow:],
            weights=nocut_count[n_underflow:],
            histtype="step",
            linestyle="dashed",
            color="k",
        )

    print(f"Saving {plot_path}")
    axis.set_xlabel(r"$\Delta M$ /MeV")
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _dataframes(
    year: str,
    magnetisation: str,
    sign: str,
    bdt_clf,
    bdt_cut,
):
    """
    The right dataframes

    """
    return cut_dfs(
        get.data(year, sign, magnetisation),
        bdt_clf,
        perform_cut=bdt_cut,
    )


def main():
    """
    Do mass fits in each time bin

    """
    year, magnetisation, sign = "2018", "magdown", "dcs"

    low, high = pdfs.domain()
    fit_range = pdfs.reduced_domain()
    n_underflow = 3
    mass_bins = definitions.nonuniform_mass_bins(
        (low, fit_range[0], high), (n_underflow, 250)
    )

    # Get the classifier for BDT cut if we need
    bdt_clf = get_clf(year, "dcs", magnetisation)

    # Get the efficiency reweighters if we need
    # Get generators of dataframes
    cut_dfs = _dataframes(year, magnetisation, sign, bdt_clf, bdt_cut=True)
    nocut_dfs = _dataframes(year, magnetisation, sign, None, bdt_cut=False)

    # Find counts
    cut_count, cut_err = mass_util.delta_m_counts(cut_dfs, mass_bins)
    nocut_count, nocut_err = mass_util.delta_m_counts(nocut_dfs, mass_bins)

    # Plot with the BDT cut
    _separate_fit(nocut_count, nocut_err, mass_bins, n_underflow, "showpeak_nocut.png")

    # Plot without the BDT cut
    _separate_fit(
        cut_count,
        cut_err,
        mass_bins,
        n_underflow,
        "showpeak_cut.png",
        nocut_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WS mass fit plot")

    main(**vars(parser.parse_args()))
