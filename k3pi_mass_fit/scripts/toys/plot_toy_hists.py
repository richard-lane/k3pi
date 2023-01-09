"""
Plot histograms of toys

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))

from libFit import toy_utils, pdfs


def main():
    """
    Generate and plot with/without alt background model

    """
    rng = np.random.default_rng(seed=0)
    n_sig, n_bkg = 1000000, 1000000
    sign, time_bin = "dcs", 5
    bkg_kw = {"n_bins": 100, "sign": sign, "bdt_cut": False, "efficiency": False}

    pts, _ = toy_utils.gen_points(
        rng, n_sig, n_bkg, sign, time_bin, pdfs.background_defaults(sign)
    )
    alt_pts, _ = toy_utils.gen_points(
        rng, n_sig, n_bkg, sign, time_bin, (0.0, 0.0, 0.0), bkg_kw
    )

    plot_kw = {"bins": np.linspace(*pdfs.domain(), 100), "histtype": "step"}
    plt.hist(pts, **plot_kw, label="sqrt bkg")
    plt.hist(alt_pts, **plot_kw, label="alt bkg")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
