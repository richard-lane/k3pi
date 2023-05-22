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
from lib_data import get, ipchi2_fit, cuts
from lib_time_fit import definitions


def _ipchi2s(
    year: str, sign: str, magnetisation: str, time_range: Tuple[float, float]
) -> np.ndarray:
    """
    ipchi2 in the given time range

    """
    # Make it run faster; set to None for all
    n_dfs = None

    low_t, high_t = time_range
    dfs = (
        dataframe[(low_t < dataframe["time"]) & (dataframe["time"] < high_t)]
        for dataframe in cuts.cands_cut_dfs(
            islice(get.data(year, sign, magnetisation), n_dfs)
        )
    )
    return np.concatenate([np.log(dataframe["D0 ipchi2"]) for dataframe in dfs])


def main(*, year: str, sign: str, magnetisation: str):
    """
    Get the D0 ipchi2 counts from the data
    Fit to them + plot

    """
    low_ip, high_ip = ipchi2_fit.domain()

    # Only want the second+ time bins, since we fixed the shape
    # of the prompt peak from the first time bin
    time_bins = definitions.TIME_BINS[2:-1]

    # Get the fit params from the peak in the first time bin
    sig_defaults = dict(
        zip(
            ["centre", "width_l", "width_r", "alpha_l", "alpha_r", "beta"],
            ipchi2_fit.low_t_params(sign),
        )
    )
    bkg_defaults = {
        "centre": 4.5,
        "width": 1.8,
        "alpha": 0.00,
        "beta": 0.00,
    }

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
    ]
    centre_limits = (
        (3.5, 4.5),
        (3.5, 4.5),
        (3.5, 4.5),
        (3.5, 4.5),
        (3.5, 5.0),
        (3.5, 5.0),
        (3.5, 5.0),
        (3.5, 5.0),
        (3.5, 6.0),
    )
    width_limits = (
        (0.4, 1.5),
        (0.4, 1.5),
        (0.4, 1.5),
        (0.4, 1.5),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.5, 1.5),
    )
    sec_fracs = []
    for i, (low_t, high_t, frac_guess, centre_lims, width_lims) in tqdm(
        enumerate(
            zip(
                time_bins[:-1],
                time_bins[1:],
                sec_frac_guesses,
                centre_limits,
                width_limits,
            )
        )
    ):
        # Get counts in the bins
        ipchi2s = _ipchi2s(year, sign, magnetisation, (low_t, high_t))[:750_000]

        # Get rid of points outside domain where the PDF is defined
        low_ip, high_ip = ipchi2_fit.domain()
        ipchi2s = ipchi2s[ipchi2s > low_ip]
        ipchi2s = ipchi2s[ipchi2s < high_ip]

        # fit
        fitter = ipchi2_fit.fixed_prompt_unbinned_fit(
            ipchi2s, 1 - frac_guess, sig_defaults, bkg_defaults, centre_lims, width_lims
        )
        print(fitter)

        # plot
        fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(10, 12))
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
        axes["A"].axvline(x=np.log(cuts.MAX_IPCHI2), color="r")

        # Append the secondary fraction (below the cut) to the list
        sec_frac = ipchi2_fit.sec_frac_below_cut(sig_defaults, fitter.values)
        sec_fracs.append(sec_frac)

        fig.suptitle(
            f"${low_t} < \\frac{{t}}{{\\tau}} < {high_t}$\n$f_\mathrm{{sec}}=${100 * sec_frac:.2f}%\n{fitter.valid=}"
        )
        axes["A"].set_title(
            str(
                dict(zip(fitter.parameters, [f"{float(x):.3f}" for x in fitter.values]))
            )
            + f"\n{fitter.valid=}"
        )

        fig.tight_layout()

        path = f"{sign}_data_ipchi2_fit_{i}.png"
        print(f"plotting {path}")
        fig.savefig(path)
        plt.close(fig)

        with open(f"plot_pkls/{path}.pkl", "wb") as f:
            pickle.dump((fig, axes), f)

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
    with open(f"plot_pkls/{path}.pkl", "wb") as f:
        pickle.dump((fig, axes), f)

    # Dump secondary fraction to file
    with open(str(ipchi2_fit.sec_frac_file(sign)), "wb") as dump_f:
        pickle.dump(sec_fracs, dump_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform IPCHI2 fit to measure secondary fraction"
    )

    parser.add_argument(
        "year", type=str, choices={"2016", "2017", "2018"}, help="data taking year"
    )
    parser.add_argument("sign", type=str, choices={"dcs", "cf"}, help="data type")
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magup", "magdown"},
        help="magnetisation direction",
    )

    main(**vars(parser.parse_args()))
