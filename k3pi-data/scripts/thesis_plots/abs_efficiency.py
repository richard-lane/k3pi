"""
Plot particle-gun derived absolute efficiency

"""
import sys
import pathlib

import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))

from lib_data import get


def main():
    """
    Plot correlation matrices

    """
    year, magnetisation = "2018", "magup"
    # Convert to percent
    dcs_eff = 100 * get.absolute_efficiency(year, "dcs", magnetisation)
    cf_eff = 100 * get.absolute_efficiency(year, "cf", magnetisation)

    dcs_err = 100 * get.abs_eff_err(year, "dcs", magnetisation)
    cf_err = 100 * get.abs_eff_err(year, "cf", magnetisation)

    fig, axis = plt.subplots()

    axis.errorbar(-1, dcs_eff, yerr=dcs_err, fmt="k.")
    axis.errorbar(1, cf_eff, yerr=cf_err, fmt="k.")

    xlim = 2
    axis.set_xticks([-1, 1])
    axis.set_xlim([-xlim, xlim])
    axis.set_xticklabels(["DCS", "CF"])

    axis.set_ylabel("Efficiency/ %")
    # axis.set_ylim(2.23, 2.285)

    # Draw a line from the error bars to the axes to make it easier to read
    axis.plot([-xlim, -1.0], [dcs_eff - dcs_err, dcs_eff - dcs_err], "k--")
    axis.plot([-xlim, -1.0], [dcs_eff + dcs_err, dcs_eff + dcs_err], "k--")

    axis.plot([-xlim, 1.0], [cf_eff - cf_err, cf_eff - cf_err], "k--")
    axis.plot([-xlim, 1.0], [cf_eff + cf_err, cf_eff + cf_err], "k--")

    axis.text(-xlim, dcs_eff - 0.5 * dcs_err, "DCS")
    axis.text(-xlim, cf_eff - 0.5 * cf_err, "CF")

    fig.tight_layout()

    fig.savefig(f"abs_efficiency_{year}_{magnetisation}.png")


if __name__ == "__main__":
    main()
