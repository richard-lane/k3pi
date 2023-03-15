"""
Fit to D0 IPCHI2 histogram

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_fitter"))
from lib_data import get, ipchi2_fit
from lib_time_fit import definitions


def _ipchi2s(sign: str, time_range: Tuple[float, float]) -> np.ndarray:
    """
    ipchi2 in the given time range

    """
    low_t, high_t = time_range
    dfs = (
        dataframe[(low_t < dataframe["time"]) & (dataframe["time"] < high_t)]
        for dataframe in get.data("2018", sign, "magdown")
    )
    return np.concatenate([np.log(dataframe["D0 ipchi2"]) for dataframe in dfs])


def main(*, sign: str):
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    low_ip, high_ip = ipchi2_fit.domain()
    time_bins = definitions.TIME_BINS
    time_bins = [0.0, 19.0]

    sec_frac_guesses = [
        0.0,
        0.05,
        0.05,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.45,
        0.75,
        0.95,
    ]
    sec_frac_guesses = [0.4]
    for i, (low_t, high_t, frac_guess) in tqdm(
        enumerate(zip(time_bins[:-1], time_bins[1:], sec_frac_guesses))
    ):
        # Get counts in the bins
        ipchi2s = _ipchi2s(sign, (low_t, high_t))[:500000]

        # fit
        sig_defaults = {
            "centre_sig": 0.9,
            "width_l_sig": 1.6,
            "width_r_sig": 0.9,
            "alpha_l_sig": 0.0,
            "alpha_r_sig": 0.0,
            "beta_sig": 0.0,
        }
        bkg_defaults = {
            "centre_bkg": 4.5,
            "width_bkg": 1.8,
            "alpha_bkg": 0.0,
            "beta_bkg": 0.0,
        }
        fitter = ipchi2_fit.unbinned_fit(
            ipchi2s, 1 - frac_guess, sig_defaults, bkg_defaults
        )

        # plot
        fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(8, 10))
        ip_bins = np.linspace(low_ip, high_ip, 250)
        counts, _ = np.histogram(ipchi2s, ip_bins)
        ipchi2_fit.plot(
            (axes["A"], axes["B"]), ip_bins, counts, np.sqrt(counts), fitter.values
        )
        axes["A"].legend()

        path = f"{sign}_data_ipchi2_fit_{i}.png"
        print(f"plotting {path}")
        f_sec = (
            100
            * fitter.values["n_bkg"]
            / (fitter.values["n_sig"] + fitter.values["n_bkg"])
        )
        fig.suptitle(
            f"${low_t} < \\frac{{t}}{{\\tau}} < {high_t}$\n$f_\mathrm{{sec}}=${f_sec:.2f}%\n{fitter.valid=}"
        )
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform IPCHI2 fit to measure secondary fraction"
    )

    parser.add_argument("sign", type=str, choices={"dcs", "cf"}, help="data type")

    main(**vars(parser.parse_args()))
