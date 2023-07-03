"""
Plot the efficiency used for the toy reweighting

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_efficiency import mock_efficiency


def main():
    """
    Plot the decay time and phase space efficiencies
    as functions of their dependent variables

    """
    n = 1000
    times = np.linspace(0, 10, n)
    k_pt = np.linspace(0, 1000, n)

    # time and phsp factors
    dcs_factors = 0.98, 1.0
    cf_factors = 1.0, 0.98

    dcs_time_eff = mock_efficiency.time_efficiency(
        {"time": times}, factor=dcs_factors[0]
    )
    cf_time_eff = mock_efficiency.time_efficiency({"time": times}, factor=cf_factors[0])

    dcs_phsp_eff = mock_efficiency.kpt_eff(k_pt, factor=dcs_factors[1])
    cf_phsp_eff = mock_efficiency.kpt_eff(k_pt, factor=cf_factors[1])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    axes[0].plot(times, dcs_time_eff, label="DCS", color="r")
    axes[0].plot(times, cf_time_eff, label="CF", color="b")

    axes[1].plot(k_pt, dcs_phsp_eff, label="DCS", color="r")
    axes[1].plot(k_pt, cf_phsp_eff, label="CF", color="b")

    axes[0].legend()

    axes[0].set_xlabel(r"time/ $\tau$")
    axes[1].set_xlabel(r"$p_T(K)$/ MeV")

    axes[0].set_ylabel("Efficiency")

    fig.tight_layout()

    fig.savefig("mock_eff.png")


if __name__ == "__main__":
    main()
