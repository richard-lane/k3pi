"""
Plot PID efficiency "histograms", using boost-histograms that were
created with pidcalib2.

Commands to create these are in k3pi-data/README.md, probably

"""
import pickle
import pathlib
import matplotlib.pyplot as plt
import boost_histogram as bh


def _2d_hist(ax: plt.Axes, histogram: bh.Histogram):
    """
    Plot 2d boost histogram on an axis

    returns something

    """
    return ax.pcolormesh(*histogram.axes.edges.T, histogram.values().T, cmap="viridis")


def main():
    """
    Read histograms from the pidcalib output dir and plot them

    """
    out_dir = pathlib.Path(__file__).resolve().parents[2] / "pidcalib_output"

    for dump in out_dir.glob("*"):
        with open(dump, "rb") as pkl:
            hist = pickle.load(pkl)
        fig, ax = plt.subplots()

        mesh = _2d_hist(ax, hist)
        fig.colorbar(mesh)

        fig.suptitle(dump)

        # These might be wrong if you used a different binning/different variables
        ax.set_xlabel("P")
        ax.set_ylabel(r"$\eta$")

        plt.show()


if __name__ == "__main__":
    main()
