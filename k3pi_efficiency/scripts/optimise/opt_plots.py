"""
Make plots using the result of the optimisation

NB the output files from the Condor jobs (e.g. output.${JOB_ID}.out) should be moved to
a directory called `output/`

"""
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


def _good_file(filepath: str, sign: str, k_sign) -> bool:
    """
    Whether a file contains analysis of the type of reweighter we want (K+/K-; cf/dcs)

    """
    with open(filepath, "r") as f:
        header_line = f.readline().strip()

    header_line = header_line.split("\t")

    return (sign in header_line[0]) and (k_sign in header_line[1])


def _good_line(line: str) -> re.Match:
    """
    Whether a string contains the information about the BDT hyperparams

    """
    # n_estimators is an int
    # max_depth is an int < 10
    # learning rate is either a float like 0.123123123 or 4.123123e-2
    # min_samples_leaf is an int
    # distance and area are floats like the learning rate, except they may not start with 0
    pattern = (
        r"^n_estimators\t(\d+)\tmax_depth\t(\d)\tlearning_rate\t(\d\.\d+e-\d+|0\.\d+)"
        r"\tmin_samples_leaf\t(\d+)\t(\d\.\d+e-\d+|\d\.\d+)\t(\d\.\d+e-\d+|\d\.\d+)"
    )
    regex = re.compile(pattern)
    return regex.match(line)


def _params(
    filepath: str,
) -> np.ndarray:
    """
    tuple of arrays of n_estimators, max_depth, learning_rate, min_samples_leaf, distance, area
    from a file

    Returns a (6, N) shape array holding parameters

    """
    # This does mean we're opening files twice but that's fine
    retval = tuple(np.empty(0) for _ in range(6))

    with open(filepath, "r") as f:
        for line in f.readlines():
            match = _good_line(line)
            if match:
                values = np.array([float(val) for val in match.groups()])

                # Use np.c_ to reshape + stack the params into the right shape
                # Effectively this just takes sticks our 6-length tuple onto the
                # end of our (6, N) shape array of params in the right way
                retval = np.c_[retval, values]
            else:
                assert (
                    "n_estimators" not in line
                ), f"{line}\nmissed by regex"  # Catch regex bugs

    return retval


def _mins(params: np.ndarray, n_vals: int) -> np.ndarray:
    """
    Find the parameters for which the hull distances are minimal

    """
    min_indices = np.argpartition(params[4], n_vals)[:n_vals]

    return params[:, min_indices]


def _plots(params: np.ndarray, n_mins: int) -> None:
    """
    Make plots using the params

    Plot also faded red circles around the lowest n_mins vals

    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharey=True)

    # Red circles around minima
    min_params = _mins(params, n_mins)
    scatter_kw = {"y": min_params[4], "c": "r", "s": 15.0, "alpha": 0.5}
    ax[0, 0].scatter(min_params[0], **scatter_kw)
    ax[0, 1].scatter(min_params[1], **scatter_kw)
    ax[1, 0].scatter(min_params[2], **scatter_kw)
    ax[1, 1].scatter(min_params[3], **scatter_kw)

    distances = params[4]
    areas = params[5]
    colourscale = LogNorm(vmin=np.min(areas), vmax=np.max(areas))

    scatter_kw = {
        "y": distances,
        "c": colourscale(areas),
        "s": 1.5,
        "alpha": None,
        "cmap": "cividis",
    }
    ax[0, 0].scatter(params[0], **scatter_kw)
    ax[0, 1].scatter(params[1], **scatter_kw)
    ax[1, 0].scatter(params[2], **scatter_kw)
    ax[1, 1].scatter(params[3], **scatter_kw)

    titles = "n estimators", "max depth", "learning rate", "min samples per leaf"

    for axis, title in zip(ax.ravel(), titles):
        axis.set_yscale("log")
        axis.set_title(title)

    for axis in ax[:, 0]:
        axis.set_ylabel(r"AmpGen $\rightarrow$ Reweighted Hull Distance")

    fig.tight_layout(rect=[0, 0, 0.87, 1])

    cbar_ax = fig.add_axes([0.875, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(
        ScalarMappable(norm=colourscale, cmap=scatter_kw["cmap"]),
        cax=cbar_ax,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Reweighted Hull Area", labelpad=0.1)


def main(sign: str, k_sign: str):
    """
    For output files matching the sign/k_sign we want, use the numbers within to make plots of the
    optimisation results

    """
    # Array of (n_estimators, max_depth, learning_rate, min_samples_leaf, distance, area)
    params = np.empty((6, 0))

    # Open all the output files
    for filepath in glob.glob("output/*"):
        if _good_file(filepath, sign, k_sign):
            params = np.column_stack((params, _params(filepath)))

    # Make plots
    n_mins = 5
    _plots(params, n_mins)

    # Print the minimum params
    for array, header in zip(
        _mins(params, n_mins)[:4],
        ("n estimators", "max depth", "learning rate", "min per leaf"),
    ):
        print(f"{header}\t" + "\t".join(f"{x:<5f}" for x in array))

    plt.savefig(f"opt_{sign}_{k_sign}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the result of the output files from the optimisation condor jobs."
        "Must put these files in `output/`"
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="type of decay, CF or DCS",
    )
    parser.add_argument(
        "k_sign",
        type=str,
        choices={"k_plus", "k_minus"},
        help="whether to read K+ or K- data",
    )

    args = parser.parse_args()
    main(
        args.sign,
        args.k_sign,
    )
