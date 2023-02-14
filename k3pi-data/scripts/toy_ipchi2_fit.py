"""
Generate some toy IPCHI2 evts according to the model,
fit the model back

"""
import sys
import pathlib
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import ipchi2_fit, stats


def _generate(rng: np.random.Generator, n_gen: int, params: Tuple) -> np.ndarray:
    """
    Generate events according to the peak model

    """
    low, high = ipchi2_fit.domain()
    max_ = 1.05 * np.max(ipchi2_fit.norm_peak(np.linspace(low, high, 100), *params))

    x = low + (high - low) * rng.random(n_gen)
    y = max_ * rng.random(n_gen)
    vals = ipchi2_fit.norm_peak(x, *params)
    assert (vals < max_).all()

    return x[y < vals]


def main():
    """
    Accept-reject to generate points with the model
    Sig + bkg separately
    Then do fit
    Then plot a histogram + the fit on the same plot

    """
    rng = np.random.default_rng()
    sig_defaults = {
        "centre_sig": 0.0,
        "width_l_sig": 1.5,
        "width_r_sig": 1.5,
        "alpha_l_sig": 0.1,
        "alpha_r_sig": 0.1,
        "beta_sig": 0.0,
    }
    bkg_defaults = {
        "centre_bkg": 7.0,
        "width_l_bkg": 3.0,
        "width_r_bkg": 2.5,
        "alpha_l_bkg": 0.1,
        "alpha_r_bkg": 0.1,
        "beta_bkg": 0.0,
    }
    sig = _generate(rng, 750_000, tuple(sig_defaults.values()))
    bkg = _generate(rng, 100_000, tuple(bkg_defaults.values()))

    bins = np.unique(
        np.concatenate(
            (
                np.linspace(ipchi2_fit.domain()[0], -5, 10),
                np.linspace(-5, 12, 50),
                np.linspace(12, ipchi2_fit.domain()[1], 10),
            )
        )
    )
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = (bins[1:] - bins[:-1]) / 2

    sig_counts, _ = stats.counts(sig, bins)
    bkg_counts, _ = stats.counts(bkg, bins)

    counts = sig_counts + bkg_counts

    fitter = ipchi2_fit.fit(counts, bins, 0.6, sig_defaults, bkg_defaults)

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(8, 10))
    ipchi2_fit.plot(
        (axes["A"], axes["B"]), bins, counts, np.sqrt(counts), fitter.values
    )

    axes["A"].errorbar(
        centres, sig_counts, xerr=bin_widths, yerr=np.sqrt(sig_counts), fmt="r."
    )
    axes["A"].errorbar(
        centres, bkg_counts, xerr=bin_widths, yerr=np.sqrt(bkg_counts), fmt="r."
    )

    plt.tight_layout()
    plt.savefig("toy_ipchi2_fit.png")
    plt.show()


if __name__ == "__main__":
    main()
