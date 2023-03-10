"""
Show the RS and WS MC signal peaks

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from lib_data import get
from libFit import util, definitions, pdfs


def _density(sign: str):
    """
    delta M density in the bins

    """
    fit_low, _ = pdfs.reduced_domain()
    gen_low, gen_high = pdfs.domain()
    n_underflow = 3
    bins = definitions.nonuniform_mass_bins(
        (gen_low, fit_low, 144.0, 147.0, gen_high), (n_underflow, 30, 50, 30)
    )

    count, _ = np.histogram(util.delta_m(get.mc("2018", sign, "magdown")), bins)
    print(f"{sign}\t{np.sum(count)}")

    widths = bins[1:] - bins[:-1]

    return count / widths, bins


def main():
    """
    Get the MC and plot them as histograms

    """
    rs_density, bins = _density("cf")
    ws_density, _ = _density("dcs")

    fig, axis = plt.subplots(figsize=(5, 5), sharex=True, sharey=True)

    hist_kw = {"histtype": "step", "bins": bins, "x": bins[1:]}
    axis.hist(**hist_kw, weights=rs_density, label="CF", color="r")
    axis.hist(**hist_kw, weights=ws_density, label="DCS", color="b", linestyle="--")

    axis.set_label(r"$\Delta M$ /MeV")
    axis.set_ylabel("count /MeV")

    axis.legend()

    fig.tight_layout()
    fig.savefig("mc_peaks.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WS mass fit plot")

    main(**vars(parser.parse_args()))
