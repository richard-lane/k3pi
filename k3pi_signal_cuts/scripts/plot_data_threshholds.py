"""
Plot delta M in data at various signal significances

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts.get import classifier as get_clf
from lib_cuts import definitions
from lib_data import get, training_vars


def _delta_m(data: pd.DataFrame) -> np.ndarray:
    """D* - D0 Mass"""
    return data["D* mass"] - data["D0 mass"]


def _plot(axis: plt.Axes, delta_m: np.ndarray, threshhold: float) -> None:
    """
    Plot delta M distribution on an axis

    """
    bins = np.linspace(139, 152, 120)
    counts, _ = np.histogram(delta_m, bins)
    centres = (bins[:-1] + bins[1:]) / 2
    axis.plot(centres, counts, label=f"{threshhold:.3f}")


def main():
    """
    Plot DCS delta M for various values of the cut threshhold

    For each, work out the signal significance

    """
    year, magnetisation = "2018", "magdown"
    ws_df = pd.concat(get.data(year, "dcs", magnetisation))
    rs_df = pd.concat(get.data(year, "cf", magnetisation))

    # Use the DCS classifier
    clf = get_clf(year, "dcs", magnetisation)
    training_var_names = list(training_vars.training_var_names())
    ws_sig_probs = clf.predict_proba(ws_df[training_var_names])[:, 1]
    rs_sig_probs = clf.predict_proba(rs_df[training_var_names])[:, 1]

    # For various values of the threshhold, perform cuts
    # and plot the resultant delta M distribution
    threshholds = [0.0, 0.05, definitions.THRESHOLD, 0.30, 0.50, 0.80, 1.0]

    ws_delta_m = _delta_m(ws_df)
    rs_delta_m = _delta_m(rs_df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for threshhold in threshholds:
        _plot(axes[0], ws_delta_m[ws_sig_probs > threshhold], threshhold)
        _plot(axes[1], rs_delta_m[rs_sig_probs > threshhold], threshhold)

    fig.suptitle("BDT cuts at various probability threshholds")
    axes[0].legend(title="Threshhold")

    axes[0].set_title("WS")
    axes[1].set_title("RS")

    for axis in axes:
        axis.set_xlabel(r"$\Delta M$ /MeV")

    axes[0].text(
        146.2,
        4000,
        "optimal significance\n(in simulation)",
        color="green",
        fontsize=8,
        rotation=3,
    )

    fig.savefig("data_threshholds.png")


if __name__ == "__main__":
    main()
