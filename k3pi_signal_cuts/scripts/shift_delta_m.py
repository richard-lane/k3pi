"""
See how changing the Z momentum of daughter particles affects delta M,
to check if our BDT is sneakily working out delta M and using that
to cut on

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from lib_data import get
from lib_cuts.get import classifier as get_clf
from lib_time_fit.util import ratio_err


def _inv_mass(*arrays):
    """
    Invariant masses

    Each array should be (px, py, pz, E)

    """
    vector_sum = np.sum(arrays, axis=0)

    return np.sqrt(vector_sum[3] ** 2 - np.sum(vector_sum[:3] ** 2, axis=0))


def _shift(particle: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Shift a particles z momentum by scale_factor (and the energy by the right amount)

    """
    new_momenta = np.array([particle[0], particle[1], scale_factor * particle[2]])
    mass = _inv_mass(particle)
    new_energy = np.sqrt(mass**2 + np.sum(new_momenta**2, axis=0))

    return np.row_stack((*new_momenta, new_energy))


def _shift_particles(
    scale_factor: float, *particles: Tuple[np.ndarray, ...]
) -> Tuple[np.ndarray, ...]:
    """
    Shift momenta of particles

    """
    return tuple(_shift(particle, scale_factor) for particle in particles)


def _daughters(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    K 3pi / slow pi arrays

    """
    suffices = "Px", "Py", "Pz", "E"
    k = np.row_stack([dataframe[f"Kplus_{s}"] for s in suffices])
    pi1 = np.row_stack([dataframe[f"pi1minus_{s}"] for s in suffices])
    pi2 = np.row_stack([dataframe[f"pi2minus_{s}"] for s in suffices])
    pi3 = np.row_stack([dataframe[f"pi3plus_{s}"] for s in suffices])
    slow_pi = np.row_stack([dataframe[f"slowpi_{s}"] for s in suffices])

    return k, pi1, pi2, pi3, slow_pi


def _delta_m(k, pi1, pi2, pi3, slow_pi):
    """
    Find delta M from daughters

    """
    return _inv_mass(k, pi1, pi2, pi3, slow_pi) - _inv_mass(k, pi1, pi2, pi3)


def _bdt_df(dataframe: pd.DataFrame, shift: float) -> pd.DataFrame:
    """
    Create a dataframe containing the BDT training variables and delta M

    """
    # Create an empty dataframe
    bdt_dataframe = pd.DataFrame()

    # First few columns don't depend on momentum
    bdt_dataframe[r"ReFit $\chi^2$"] = dataframe[r"ReFit $\chi^2$"]
    bdt_dataframe[r"D0 End Vtx $\chi^2$"] = dataframe[r"D0 End Vtx $\chi^2$"]
    bdt_dataframe[r"D0 Origin Vtx $\chi^2$"] = dataframe[r"D0 Origin Vtx $\chi^2$"]
    bdt_dataframe[r"$\pi_s$ ProbNN$\pi$"] = dataframe[r"$\pi_s$ ProbNN$\pi$"]

    # Recalculate the columns that do depend on momentum
    k, pi1, pi2, pi3, slow_pi = _shift_particles(shift, *_daughters(dataframe))

    def p_t(arr: np.ndarray) -> np.ndarray:
        """transverse momentum"""
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2)

    daughter_pt = tuple(p_t(particle) for particle in (k, pi1, pi2, pi3))

    # D0 pT
    d0_momentum = k[:3] + pi1[:3] + pi2[:3] + pi3[:3]
    bdt_dataframe[r"D0 $p_T$"] = p_t(d0_momentum)

    bdt_dataframe[r"$\pi_s$ $p_T$"] = p_t(slow_pi)
    bdt_dataframe[r"$\pi_s$ IP$\chi^2$"] = dataframe[r"$\pi_s$ IP$\chi^2$"]

    # 3pi vars (min, max, sum pT)
    bdt_dataframe[r"$3\pi$ max $p_T$"] = np.amax(daughter_pt[1:], axis=0)
    bdt_dataframe[r"$3\pi$ min $p_T$"] = np.amin(daughter_pt[1:], axis=0)
    bdt_dataframe[r"$3\pi$ sum $p_T$"] = np.sum(daughter_pt[1:], axis=0)

    # k3pi vars (min, max, sum pT)
    bdt_dataframe[r"$K3\pi$ max $p_T$"] = np.amax(daughter_pt, axis=0)
    bdt_dataframe[r"$K3\pi$ min $p_T$"] = np.amin(daughter_pt, axis=0)
    bdt_dataframe[r"$K3\pi$ sum $p_T$"] = np.sum(daughter_pt, axis=0)

    # Add delta M
    bdt_dataframe["delta M"] = _delta_m(k, pi1, pi2, pi3, slow_pi)

    return bdt_dataframe


def _plot_hists(axis: plt.Axes, bins: np.ndarray, dfs, colours, label):
    """
    Plot stuff on an axis
    """
    # Keep track of how many we keep in each bin
    counts = [None] * 4

    hist_kw = {"density": False, "histtype": "step", "alpha": 0.5}
    # Plot hists
    for i, (dataframe, colour) in enumerate(zip(dfs, colours)):
        counts[i], _, _ = axis.hist(
            dataframe[label], bins=bins, **hist_kw, color=colour
        )

    # Plot efficiency
    _, _, shifted_count, kept_count = counts
    centres = (bins[1:] + bins[:-1]) / 2
    efficiency, err = ratio_err(
        kept_count, shifted_count, np.sqrt(kept_count), np.sqrt(shifted_count)
    )

    axis.twinx().errorbar(centres, efficiency, yerr=err, fmt="k.")


def _legend(ax, colours, labels):
    """legend"""
    ax.legend(
        handles=[
            Patch(color=colour, label=label, alpha=0.5)
            for colour, label in zip(colours, labels)
        ]
        + [Line2D([0], [0], marker=".", color="k", label="Cut Efficiency")]
    )


def _plot(
    uppermass: pd.DataFrame,
    data: pd.DataFrame,
    shifted_df: pd.DataFrame,
    predicted_signal: np.ndarray,
) -> None:
    """
    Plot histograms of training vars + also the cut efficiency for the shifted dataframe

    """
    # plot histograms of these shifted variables
    n_bins = 50
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning

    fig, ax = plt.subplots(5, 3, figsize=(15, 9))
    delta_m_fig, delta_m_ax = plt.subplots(figsize=(12, 8))

    dfs = (data, uppermass, shifted_df, shifted_df.loc[predicted_signal])
    labels = ("Data", "Bkg", "Bkg (Shifted)", "Bkg (Shifted), after cuts")
    colours = ("red", "orange", "blue", "green")

    for label, axis in zip(shifted_df, ax.ravel()):
        # Histogram bins
        uppermass_quantile = np.quantile(uppermass[label], quantiles)
        data_quantile = np.quantile(data[label], quantiles)
        shifted_quantile = np.quantile(shifted_df[label], quantiles)
        bins = np.linspace(
            min(uppermass_quantile[0], data_quantile[0], shifted_quantile[0]),
            max(uppermass_quantile[1], data_quantile[1], shifted_quantile[1]),
            n_bins,
        )

        # Plot the big plot
        _plot_hists(axis, bins, dfs, colours, label)

        # Also plot delta m on its own thing
        if "delta" in label:
            _plot_hists(delta_m_ax, bins, dfs, colours, label)

        axis.set_title(label)

    delta_m_ax.set_xlabel(r"$\Delta M$")

    # Legends
    _legend(ax.ravel()[-1], colours, labels)
    _legend(delta_m_ax, colours, labels)

    fig.tight_layout()
    fig.savefig("shifted_vars.png")
    delta_m_fig.tight_layout()
    delta_m_fig.savefig("shifted_deltam.png")
    plt.show()


def main():
    """
    Scale the Z momentum of upper mass sideband data (bkg) to push
    these (bkg) events into the signal region - then see the effect
    of the BDT cut on the shifted data

    NB we work out delta M here by using the daughter particles,
    not by using the D0 and Dst masses from the ROOT files - this will gives
    slightly different delta M, but it shouldn't be too much of a problem

    """
    # Read momenta
    uppermass = pd.concat(get.uppermass("2018", "dcs", "magdown"))
    data = pd.concat(get.data("2018", "dcs", "magdown"))

    shift = 0.04
    shifted_df = _bdt_df(uppermass, shift)

    uppermass["delta M"] = _delta_m(*_daughters(uppermass))
    data["delta M"] = _delta_m(*_daughters(data))

    # Classify
    clf = get_clf("2018", "dcs", "magdown")
    shifted_probs = clf.predict_proba(
        shifted_df.loc[:, shifted_df.columns != "delta M"]
    )[:, 1]
    predicted_signal = shifted_probs > 0.2

    _plot(uppermass, data, shifted_df, predicted_signal)


if __name__ == "__main__":
    main()
