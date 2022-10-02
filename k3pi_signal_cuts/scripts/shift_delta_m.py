"""
Test script to see how changing the momenta affects delta M

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import get
from lib_cuts.get import classifier as get_clf


def _inv_mass(*arrays):
    """
    Invariant masses

    Each array should be (px, py, pz, E)

    """
    vector_sum = np.sum(arrays, axis=0)

    return np.sqrt(
        vector_sum[3] ** 2
        - vector_sum[0] ** 2
        - vector_sum[1] ** 2
        - vector_sum[2] ** 2
    )


def _shift(particle: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Shift a particles 3 momentum by scale_factor (and the energy by the right amount)

    """
    new_momenta = scale_factor * particle[0:3]

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


def _delta_m(dataframe: pd.DataFrame, scale_factor: float) -> pd.Series:
    """
    Scale all the relevant momenta by scale_factor, find resultant delta_m

    """
    k, pi1, pi2, pi3, slow_pi = _daughters(dataframe)

    return _inv_mass(
        *_shift_particles(scale_factor, k, pi1, pi2, pi3, slow_pi)
    ) - _inv_mass(*_shift_particles(scale_factor, k, pi1, pi2, pi3))


def _bdt_df(dataframe: pd.DataFrame, shift: float) -> pd.DataFrame:
    """
    Create a dataframe containing the BDT training variables and delta M

    """
    # Create an empty dataframe
    bdt_dataframe = pd.DataFrame()

    # Recalculate the columns that do depend on momentum
    k, pi1, pi2, pi3, slow_pi = _shift_particles(shift, *_daughters(dataframe))

    # Calculate D0 and D* energy and momenta
    d0_mass = 1864.84
    d0_momentum = k[:3] + pi1[:3] + pi2[:3] + pi3[:3]
    d0_energy = np.sqrt(
        d0_mass**2 + d0_momentum[0] ** 2 + d0_momentum[1] ** 2 + d0_momentum[2] ** 2
    )

    def p_t(arr: np.ndarray) -> np.ndarray:
        """transverse momentum"""
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2)

    k_pt = p_t(k)
    pi1_pt = p_t(pi1)
    pi2_pt = p_t(pi2)
    pi3_pt = p_t(pi3)

    # First few columns don't depend on momentum
    bdt_dataframe[r"ReFit $\chi^2$"] = dataframe[r"ReFit $\chi^2$"]
    bdt_dataframe[r"D0 End Vtx $\chi^2$"] = dataframe[r"D0 End Vtx $\chi^2$"]
    bdt_dataframe[r"D0 Origin Vtx $\chi^2$"] = dataframe[r"D0 Origin Vtx $\chi^2$"]
    bdt_dataframe[r"$\pi_s$ ProbNN$\pi$"] = dataframe[r"D0 Origin Vtx $\chi^2$"]

    # D0 pT
    bdt_dataframe[r"D0 $p_T$"] = p_t(np.vstack((d0_momentum, d0_energy)))

    bdt_dataframe[r"$\pi_s$ $p_T$"] = p_t(slow_pi)
    bdt_dataframe[r"$\pi_s$ IP$\chi^2$"] = dataframe[r"$\pi_s$ IP$\chi^2$"]

    # 3pi vars (min, max, sum pT)
    bdt_dataframe[r"$3\pi$ max $p_T$"] = np.amax((pi1_pt, pi2_pt, pi3_pt), axis=0)
    bdt_dataframe[r"$3\pi$ min $p_T$"] = np.amin((pi1_pt, pi2_pt, pi3_pt), axis=0)
    bdt_dataframe[r"$3\pi$ sum $p_T$"] = np.sum((pi1_pt, pi2_pt, pi3_pt), axis=0)

    # k3pi vars (min, max, sum pT)
    bdt_dataframe[r"$K3\pi$ max $p_T$"] = np.amax(
        (k_pt, pi1_pt, pi2_pt, pi3_pt), axis=0
    )
    bdt_dataframe[r"$K3\pi$ min $p_T$"] = np.amin(
        (k_pt, pi1_pt, pi2_pt, pi3_pt), axis=0
    )
    bdt_dataframe[r"$K3\pi$ sum $p_T$"] = np.sum((k_pt, pi1_pt, pi2_pt, pi3_pt), axis=0)

    # Add delta M
    bdt_dataframe["delta M"] = _delta_m(dataframe, shift)

    return bdt_dataframe


def main():
    """
    Make various histograms of delta M

    """
    # Read momenta
    uppermass = pd.concat(get.uppermass("2018", "dcs", "magdown"))
    data = pd.concat(get.data("2018", "dcs", "magdown"))

    shift = 0.67
    shifted_df = _bdt_df(uppermass, shift)

    uppermass["delta M"] = uppermass["D* mass"] - uppermass["D0 mass"]
    data["delta M"] = data["D* mass"] - data["D0 mass"]

    # plot histograms of these shifted variables
    fig, ax = plt.subplots(2, 7, figsize=(21, 3))
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100
    hist_kw = {"density": True, "histtype": "step"}

    # Classify
    clf = get_clf("2018", "dcs", "magdown")
    shifted_probs = clf.predict_proba(
        shifted_df.loc[:, shifted_df.columns != "delta M"]
    )[:, 1]
    predicted_signal = shifted_probs > 0.2

    for label, axis in zip(shifted_df, ax.ravel()):
        u = uppermass[label]
        d = data[label]
        s = shifted_df[label]

        uppermass_quantile = np.quantile(u, quantiles)
        data_quantile = np.quantile(d, quantiles)
        shifted_quantile = np.quantile(s, quantiles)

        bins = np.linspace(
            min(uppermass_quantile[0], data_quantile[0], shifted_quantile[0]),
            max(uppermass_quantile[1], data_quantile[1], shifted_quantile[1]),
            n_bins,
        )

        axis.hist(uppermass[label], bins=bins, **hist_kw, label="upper mass")
        axis.hist(data[label], bins=bins, **hist_kw, label="data")
        axis.hist(shifted_df[label], bins=bins, **hist_kw, label="shifted")

        axis.hist(
            shifted_df[label][predicted_signal],
            bins=bins,
            **hist_kw,
            label="shifted, pred signal",
        )

        axis.set_title(label)

    ax[0, -1].legend()
    fig.tight_layout()
    plt.savefig("shifted_vars.png")
    plt.show()


if __name__ == "__main__":
    main()
