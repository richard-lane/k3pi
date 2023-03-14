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
    if len(params) == 6:
        print(f"gen {n_gen} prompt")
        fcn = ipchi2_fit.norm_peak
    elif len(params) == 4:
        print(f"gen {n_gen} sec")
        fcn = ipchi2_fit.secondary_peak
    else:
        raise ValueError(f"{len(params)=}")

    max_ = 1.05 * np.max(fcn(np.linspace(low, high, 100), *params))

    x = low + (high - low) * rng.random(n_gen)
    y = max_ * rng.random(n_gen)
    vals = fcn(x, *params)
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
        "centre_bkg": 5.0,
        "width_bkg": 2.0,
        "alpha_bkg": 0.1,
        "beta_bkg": 0.0,
    }
    sig = _generate(rng, 7_500_00, tuple(sig_defaults.values()))
    bkg = _generate(rng, 1_000_00, tuple(bkg_defaults.values()))

    bins = np.unique(
        np.concatenate(
            (
                np.linspace(ipchi2_fit.domain()[0], -5, 10),
                np.linspace(-5, 12, 100),
                np.linspace(12, ipchi2_fit.domain()[1], 10),
            )
        )
    )
    centres = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    sig_counts, _ = stats.counts(sig, bins)
    bkg_counts, _ = stats.counts(bkg, bins)

    counts = sig_counts + bkg_counts

    # binned_fitter = ipchi2_fit.fit(counts, bins, 0.6, sig_defaults, bkg_defaults)
    unbinned_fitter = ipchi2_fit.unbinned_fit(
        np.concatenate((sig, bkg)),
        len(sig) / (len(sig) + len(bkg)),
        sig_defaults,
        bkg_defaults,
    )

    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(8, 10))
    # ipchi2_fit.plot(
    #     (axes["A"], axes["B"]), bins, counts, np.sqrt(counts), binned_fitter.values
    # )
    ipchi2_fit.plot(
        (axes["A"], axes["B"]), bins, counts, np.sqrt(counts), unbinned_fitter.values
    )

    axes["A"].errorbar(
        centres,
        sig_counts / bin_widths,
        xerr=0.5 * bin_widths,
        yerr=np.sqrt(sig_counts),
        fmt="b.",
        label="Signal Counts",
    )
    axes["A"].errorbar(
        centres,
        bkg_counts / bin_widths,
        xerr=0.5 * bin_widths,
        yerr=np.sqrt(bkg_counts),
        fmt="r.",
        label="Bkg Counts",
    )
    axes["A"].legend()
    axes["A"].set_ylabel("Count / bin width")
    axes["A"].set_xlabel("IPCHI2")
    axes["A"].set_title("Toy")

    plt.tight_layout()
    plt.savefig("toy_ipchi2_fit.png")
    plt.show()


if __name__ == "__main__":
    main()
