"""
Test script to see how changing the momenta affects delta M

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import get


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
    new_energy = np.sqrt(mass ** 2 + np.sum(new_momenta ** 2, axis=0))

    return np.row_stack((*new_momenta, new_energy))


def _delta_m(dataframe: pd.DataFrame, scale_factor: float) -> pd.Series:
    """
    Scale all the relevant momenta by scale_factor, find resultant delta_m

    """
    suffices = "Px", "Py", "Pz", "E"
    k = np.row_stack([dataframe[f"Kplus_{s}"] for s in suffices])
    pi1 = np.row_stack([dataframe[f"pi1minus_{s}"] for s in suffices])
    pi2 = np.row_stack([dataframe[f"pi2minus_{s}"] for s in suffices])
    pi3 = np.row_stack([dataframe[f"pi3plus_{s}"] for s in suffices])
    slow_pi = np.row_stack([dataframe[f"slowpi_{s}"] for s in suffices])

    return _inv_mass(
        _shift(k, scale_factor),
        _shift(pi1, scale_factor),
        _shift(pi2, scale_factor),
        _shift(pi3, scale_factor),
        _shift(slow_pi, scale_factor),
    ) - _inv_mass(
        _shift(k, scale_factor),
        _shift(pi1, scale_factor),
        _shift(pi2, scale_factor),
        _shift(pi3, scale_factor),
    )


def _scale_factor(mass_shift: float) -> float:
    """
    Find the scale factor needed to shift delta M by the provided amount

    """
    return 1 + ((mass_shift - 0.01868964600122609) / 146.79447716308331)


def main():
    """
    Make various histograms of delta M

    """
    # Read momenta
    uppermass = pd.concat(get.uppermass("2018", "dcs", "magdown"))
    data = pd.concat(get.data("2018", "dcs", "magdown"))

    fig, axis = plt.subplots()
    hist_kw = {"bins": np.linspace(135, 180), "histtype": "step", "density": True}
    axis.hist(_delta_m(data, 1.0), **hist_kw)
    axis.hist(_delta_m(uppermass, 0.6), **hist_kw)

    plt.show()


if __name__ == "__main__":
    main()
