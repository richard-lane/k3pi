"""
Plot histograms of PID calibration efficiences,
generated with pidcalib2.

If you haven't created these already, it's probably easiest
to create them on lxplus with the commands in one of the
READMEs.

"""
import pickle
import pathlib
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh


def _plot(axis: plt.Axes, path: str) -> QuadMesh:
    """
    Plot histogram stored in a pickle dump on an axis

    """
    with open(path, "rb") as hist_f:
        hist = pickle.load(hist_f)
        return axis.pcolormesh(
            *hist.axes.edges.T, hist.values().T, vmin=0, vmax=1, cmap="magma"
        )


def main():
    """
    Read + plot histograms

    """
    paths = tuple(pathlib.Path("pidcalib_output/").glob("*"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    k_mesh, pi_mesh = None, None
    for path in paths:
        if "-Pi-" in str(path):
            print(f"Pi: plotting {path}")
            pi_mesh = _plot(axes[0], path)
        elif "-K-" in str(path):
            print(f"K: plotting {path}")
            k_mesh = _plot(axes[1], path)
        else:
            print(f"not plotting {path} (neither k nor pi?)")

    assert k_mesh
    assert pi_mesh

    axes[0].set_title(r"$\pi$")
    axes[1].set_title(r"K")

    for axis in axes:
        axis.set_xlabel(r"$p$")
        axis.set_ylabel(r"$\eta$")

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.90, 0.1, 0.05, 0.8])
    fig.colorbar(k_mesh, cax=cax)

    fig.savefig("pidcalib_hists.png")

    plt.show()


if __name__ == "__main__":
    main()
