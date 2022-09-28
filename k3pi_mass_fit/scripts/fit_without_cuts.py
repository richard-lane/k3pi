"""
Fit to some real data without doing BDT cuts

"""
import sys
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import uproot

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from libFit import pdfs
from libFit import fit


def _delta_m(sign: str) -> np.ndarray:
    """
    Delta M arrays

    """
    assert sign in {"RS", "WS"}

    data_dir = "/home/mh19137/Documents/HEP/data/data/tmp/*.root"
    delta_m = []
    tree_name = (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )

    for path in tqdm.tqdm(glob.glob(data_dir)):
        with uproot.open(path) as f:
            tree = f[tree_name]

            d0_mass = tree["Dst_ReFit_D0_M"].array()[:, 0]
            delta_m.append(tree["Dst_ReFit_M"].array()[:, 0] - d0_mass)

    return np.concatenate(delta_m)


def _plot(delta_m: np.ndarray, params: tuple) -> None:
    """
    Plot the fit

    """

    def fitted_pdf(x: np.ndarray) -> np.ndarray:
        return pdfs.fractional_pdf(x, *params)

    def fitted_sig(x: np.ndarray) -> np.ndarray:
        return pdfs.normalised_signal(x, *params[1:-2])

    def fitted_bkg(x: np.ndarray) -> np.ndarray:
        return pdfs.normalised_bkg(x, *params[-2:])

    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True, figsize=(9, 12))

    bins = np.linspace(*pdfs.domain(), 250)
    centres = (bins[1:] + bins[:-1]) / 2

    counts, _ = np.histogram(delta_m, bins)
    err = np.sqrt(counts)
    num_evts = len(delta_m)
    scale_factor = num_evts * (bins[1] - bins[0])

    ax["A"].errorbar(centres, counts, yerr=err, fmt="k.")
    predicted_counts = scale_factor * fitted_pdf(centres)
    ax["A"].plot(centres, predicted_counts, "r-")

    ax["A"].plot(
        centres,
        scale_factor * params[0] * fitted_sig(centres),
        color="orange",
        label="Signal",
    )
    ax["A"].plot(
        centres,
        scale_factor * (1 - params[0]) * fitted_bkg(centres),
        color="blue",
        label="Bkg",
    )
    ax["A"].legend()

    diffs = counts - predicted_counts
    ax["B"].plot(pdfs.domain(), [1, 1], "r-")
    ax["B"].errorbar(centres, diffs, yerr=err, fmt="k.")
    ax["B"].set_yticklabels([])

    fig.suptitle("real data")
    fig.savefig("real.png")
    plt.show()


def main():
    sign = "WS"
    delta_m = _delta_m(sign)
    low, high = pdfs.domain()
    mask = (delta_m < high) & (delta_m > low)
    delta_m = delta_m[mask]

    delta_m
    fitter = fit.fit(delta_m, sign, 5, 0.5)  # time bin 5 for now

    _plot(delta_m, fitter.values)


if __name__ == "__main__":
    main()
