"""
Create a histogram time weighter,
compare it to the fit time weighter

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_efficiency"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_fitter"))

from lib_efficiency import efficiency_definitions, efficiency_util
from lib_efficiency.reweighter import TimeWeighter
from lib_time_fit.definitions import TIME_BINS


def main():
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"

    # AmpGen points
    k_sign = "both"
    ag_t = efficiency_util.ampgen_df(sign, k_sign, train=True)["time"]

    # pgun points
    pgun_t = efficiency_util.pgun_df(year, magnetisation, sign, k_sign, train=True)[
        "time"
    ]

    # Create histogram division time weighter
    min_t = efficiency_definitions.MIN_TIME
    hist_weighter = TimeWeighter(min_t, fit=False, n_bins=20000, n_neighs=10)
    hist_weighter.fit(pgun_t, ag_t, np.ones_like(pgun_t))

    # Create time fit time weighter
    fit_weighter = TimeWeighter(min_t, fit=True, n_bins=None, n_neighs=None)
    fit_weighter.fit(pgun_t, ag_t, np.ones_like(pgun_t))

    # Find the average efficiency in each time bin from each,
    # and the error
    test_pts = np.linspace(min_t - 0.4, TIME_BINS[-2], 500)
    hist_eff = hist_weighter.apply_efficiency(test_pts)
    fit_eff = fit_weighter.apply_efficiency(test_pts)

    hist_eff /= np.mean(hist_eff)
    fit_eff /= np.mean(fit_eff)

    centres = []
    widths = []
    fit_effs = []
    hist_effs = []
    fit_errs = []
    hist_errs = []
    for low, high in zip(TIME_BINS[2:-2], TIME_BINS[3:-1]):
        in_bin = (low < test_pts) & (test_pts < high)

        centres.append(np.mean(test_pts[in_bin]))
        widths.append((centres[-1] - low, high - centres[-1]))

        fit_effs.append(np.mean(fit_eff[in_bin]))
        hist_effs.append(np.mean(hist_eff[in_bin]))

        fit_errs.append(np.std(fit_eff[in_bin]))
        hist_errs.append(np.std(hist_eff[in_bin]))
    widths = np.array(widths).T

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(test_pts, hist_eff, "k+", alpha=0.3)
    axis.plot(test_pts, fit_eff, "r--", alpha=0.3)

    axis.errorbar(
        centres,
        hist_effs,
        xerr=widths,
        yerr=hist_errs,
        fmt="k.",
        label=r"$\epsilon$ (Histogram)",
    )
    axis.errorbar(
        centres,
        fit_effs,
        xerr=widths,
        yerr=fit_errs,
        fmt="r.",
        label=r"$\epsilon$ (Fit)",
    )

    axis.legend()
    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"$\epsilon$")
    axis.set_xlim(0.5, axis.get_xlim()[1])
    fig.tight_layout()

    fig.savefig("compare_time_weighters.png")


if __name__ == "__main__":
    main()
