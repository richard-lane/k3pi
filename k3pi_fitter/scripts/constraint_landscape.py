"""
Plot the chi2 landscape for the Gaussian constraint on x and y

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_time_fit import models


def main():
    """
    Create arrays of x and y values,
    find the chi2 value of the Gaussian constraint at each
    and plot both 2d and 1d landscapes in each

    """
    # Parameters for evaluating the chi2 value
    means = (0.0, 2.0)
    widths = (1.0, 0.5)
    correlation = -0.3

    # Create arrays
    n_x, n_y = 50, 51
    x_vals = np.linspace(-5, 5, n_x)
    y_vals = np.linspace(-5, 5, n_y)
    x_x, y_y = np.meshgrid(x_vals, y_vals)

    # We dont need to set anything but the params for working out the constraint
    # and the bins (otherwise we error)
    chi2 = models.ConstrainedBase(
        None, None, np.array([0, 1]), means, widths, correlation
    ).constraint(x_x, y_y)
    chi2 = np.sqrt(chi2 - np.min(chi2))

    fig, axis = plt.subplot_mosaic(
        "DAAAAB\nDAAAAB\nDAAAAB\nDAAAAB\n.CCCC.", figsize=(6, 5)
    )
    # Want some of the axes to be shared
    axis["D"].get_shared_y_axes().join(axis["D"], axis["A"])
    axis["C"].get_shared_x_axes().join(axis["C"], axis["A"])

    # Plot 2d chi2 landscape
    n_levels = 5
    contours = axis["A"].contourf(x_x, y_y, chi2, levels=np.arange(n_levels))
    fig.colorbar(contours, cax=axis["B"])

    # Plot 1d projections
    axis["D"].plot(chi2.sum(axis=1) / n_x, y_vals)
    axis["C"].plot(x_vals, chi2.sum(axis=0) / n_y)

    axis["A"].set_xticks([])
    axis["A"].set_yticks([])

    axis["D"].set_xticks([])
    axis["C"].set_yticks([])

    fig.suptitle(r"Gaussian constraint on $x$ and $y$")
    axis["C"].set_xlabel(r"$x$")
    axis["D"].set_ylabel(r"$y$")
    axis["B"].set_ylabel(r"$\sigma$")

    plt.savefig("xy_constraint.png")


if __name__ == "__main__":
    main()
