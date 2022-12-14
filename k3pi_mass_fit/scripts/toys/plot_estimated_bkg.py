"""
Get a background PDF from a pickle dump

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, bkg
from lib_data import stats


def main():
    """
    Get the pdf, plot points

    """
    n_bins = 100
    dcs_pdf = bkg.pdf(n_bins, "dcs", bdt_cut=False, efficiency=False)
    cf_pdf = bkg.pdf(n_bins, "cf", bdt_cut=False, efficiency=False)

    points = np.linspace(*pdfs.domain(), 10000)[1:-1]

    plt.plot(
        points, pdfs.estimated_bkg(points, dcs_pdf, 0.01, 0.0002, 0.0001), label="DCS"
    )
    plt.plot(points, pdfs.estimated_bkg(points, cf_pdf, 0, 0, 0), label="CF")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
