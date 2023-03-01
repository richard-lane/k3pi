"""
Get a background PDF from a pickle dump

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))

from libFit import pdfs, bkg


def main(*, year: str, magnetisation: str, bdt_cut: bool):
    """
    Get the pdf, plot points

    """
    domain = (pdfs.domain()[0], 160.0)
    bins = np.linspace(*domain, 250)

    dcs_pdf = bkg.pdf(bins, year, magnetisation, "dcs", bdt_cut=bdt_cut)
    cf_pdf = bkg.pdf(bins, year, magnetisation, "cf", bdt_cut=bdt_cut)

    points = bins[1:-1]

    params = (0, 0, 0)
    fig, axis = plt.subplots()

    axis.plot(points, pdfs.estimated_bkg(points, dcs_pdf, domain, *params), label="WS")
    axis.plot(points, pdfs.estimated_bkg(points, cf_pdf, domain, *params), label="RS")

    axis.legend()
    axis.set_xlabel(r"$\Delta M$")
    axis.set_title(f"Estimated BKG pdf, {params=}")

    path = f"estimated_bkg_{year}_{magnetisation}_{bdt_cut=}.png"
    print(f"plotting {path}")
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pheno bkgs")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )
    parser.add_argument("--bdt_cut", action="store_true", help="BDT cut the data")

    main(**vars(parser.parse_args()))
