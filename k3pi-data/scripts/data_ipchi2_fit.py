"""
Fit to D0 IPCHI2 histogram

"""
import sys
import pickle
import pathlib
import argparse
from typing import Tuple
from itertools import islice

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
    # Make it run faster; set to None for all
    n_dfs = None

    low_t, high_t = time_range
    dfs = (
        dataframe[(low_t < dataframe["time"]) & (dataframe["time"] < high_t)]
        for dataframe in islice(get.data("2018", sign, "magdown"), n_dfs)
    )
    return np.concatenate([np.log(dataframe["D0 ipchi2"]) for dataframe in dfs])


def main(*, sign: str):
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    low_ip, high_ip = ipchi2_fit.domain()

    # Only want the second+ time bins, since we fixed the shape
    # of the prompt peak from the first time bin
    time_bins = definitions.TIME_BINS[2:-1]
    ip_cut = 9.0

    sec_frac_guesses = [
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
        # 0.95,
    ]
    sec_fracs = []
    for i, (low_t, high_t, frac_guess) in tqdm(
        enumerate(zip(time_bins[:-1], time_bins[1:], sec_frac_guesses))
    ):
        # Get counts in the bins
        ipchi2s = _ipchi2s(sign, (low_t, high_t))[:500000]

        # Get rid of points outside domain where the PDF is defined
        low_ip, high_ip = ipchi2_fit.domain()
        ipchi2s = ipchi2s[ipchi2s > low_ip]
        ipchi2s = ipchi2s[ipchi2s < high_ip]

        # fit
        sig_defaults = (
            {
                "centre": 0.977,
                "width_l": 1.230,
                "width_r": 0.813,
                "alpha_l": 0.254,
                "alpha_r": 0.117,
                "beta": 0.018,
            }
            if sign == "cf"
            else {
                "centre": 1.439,
                "width_l": 1.482,
                "width_r": 1.248,
                "alpha_l": 0.091,
                "alpha_r": -0.025,
                "beta": 0.001,
            }
        )
        bkg_defaults = {
            "centre": 4.5,
            "width": 1.8,
            "alpha": 0.01,
            "beta": 0.01,
        }
        fitter = ipchi2_fit.fixed_prompt_unbinned_fit(
            ipchi2s, 1 - frac_guess, sig_defaults, bkg_defaults
        )

        # plot
        fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(8, 10))
        ip_bins = np.linspace(low_ip, high_ip, 250)
        counts, _ = np.histogram(ipchi2s, ip_bins)
        ipchi2_fit.plot_fixed_prompt(
            (axes["A"], axes["B"]),
            ip_bins,
            counts,
            np.sqrt(counts),
            fitter.values,
            sig_defaults,
        )
        axes["A"].legend()
        axes["A"].axvline(x=np.log(ip_cut), color="r")

        # Append the secondary fraction (below the cut) to the list
        sec_frac = ipchi2_fit.sec_frac_below_cut(
            sig_defaults, fitter.values, np.log(ip_cut)
        )
        sec_fracs.append(sec_frac)

        fig.suptitle(
            f"${low_t} < \\frac{{t}}{{\\tau}} < {high_t}$\n$f_\mathrm{{sec}}=${100 * sec_frac:.2f}%\n{fitter.valid=}"
        )

        path = f"{sign}_data_ipchi2_fit_{i}.png"
        print(f"plotting {path}")
        fig.savefig(path)
        plt.close(fig)

    sec_fracs = np.array(sec_fracs)

    # Plot secondary fractions
    time_centres = (time_bins[1:] + time_bins[:-1]) / 2
    time_widths = (time_bins[1:] - time_bins[:-1]) / 2

    fig, axis = plt.subplots()
    axis.errorbar(time_centres, 100 * sec_fracs, xerr=time_widths, fmt="k+")

    axis.set_xlabel(r"t/$\tau$")
    axis.set_ylabel(r"$f_\mathrm{sec}$ / %")
    axis.set_title("Secondary Leakage")
    path = f"{sign}_data_ipchi2_leakage.png"
    print(f"plotting {path}")
    fig.savefig(path)

    # Dump secondary fraction to file
    with open(str(ipchi2_fit.sec_frac_file(sign)), "wb") as dump_f:
        pickle.dump(sec_fracs, dump_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform IPCHI2 fit to measure secondary fraction"
    )

    parser.add_argument("sign", type=str, choices={"dcs", "cf"}, help="data type")

    main(**vars(parser.parse_args()))
