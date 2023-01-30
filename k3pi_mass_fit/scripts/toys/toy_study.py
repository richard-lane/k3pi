"""
Toy study for mass fit

"""
import os
import sys
import time
import pathlib
import argparse
from typing import Tuple
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[3] / "k3pi-data"))

from libFit import pdfs, fit, toy_utils, definitions, bkg
from lib_data import stats


def _fit(fit_params, bkg_params, combined, fit_axis, sqrt_fit):
    """
    Plot fit on an axis

    """
    if sqrt_fit:

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.model(x, *fit_params)

    else:
        bkg_pdf = bkg.pdf(
            bkg_params["n_bins"],
            bkg_params["sign"],
            bdt_cut=bkg_params["bdt_cut"],
            efficiency=bkg_params["efficiency"],
        )

        def fitted_pdf(x: np.ndarray) -> np.ndarray:
            return pdfs.model_alt_bkg(x, *fit_params[:8], bkg_pdf, *fit_params[-3:])

    pts = definitions.mass_bins(100)
    centres = (pts[1:] + pts[:-1]) / 2
    widths = pts[1:] - pts[:-1]

    fit_axis.plot(centres, widths * fitted_pdf(centres), "r--")
    fit_axis.hist(combined, bins=pts)


def _pull(
    rng: np.random.Generator,
    n_sig: int,
    n_bkg: int,
    sign: str,
    time_bin: int,
    out_dict: dict,
    *,
    sqrt_gen: bool,
    sqrt_fit: bool,
) -> np.ndarray:
    """
    Find pulls for the fit parameters signal_fraction, centre, width, alpha, a, b

    Returns array of pulls

    """
    bkg_params = (
        {}
        if sqrt_gen
        else {"n_bins": 100, "sign": "cf", "bdt_cut": True, "efficiency": False}
    )

    # Perform fit
    combined, true_params = toy_utils.gen_points(
        rng,
        n_sig,
        n_bkg,
        sign,
        time_bin,
        pdfs.background_defaults(sign) if sqrt_gen else (0, 0, 0),
        bkg_params,
        verbose=False,
    )

    true_params = np.array(true_params)
    sig_frac = true_params[0] / (true_params[0] + true_params[1])

    bins = definitions.mass_bins(100 if sqrt_gen else bkg_params["n_bins"])
    counts, _ = stats.counts(combined, bins)
    fitter = (
        fit.binned_fit(counts, bins, sign, time_bin, sig_frac)
        if sqrt_fit
        else fit.alt_bkg_fit(
            counts,
            bins,
            sign,
            time_bin,
            sig_frac,
            bdt_cut=bkg_params["bdt_cut"],
            efficiency=bkg_params["efficiency"],
        )
    )

    fit_params = fitter.values
    fit_errs = fitter.errors

    if not out_dict:
        out_dict["fit_params"] = np.array(fit_params)
        out_dict["bkg_params"] = bkg_params
        out_dict["combined"] = combined

    if sqrt_gen != sqrt_fit:
        # Fitting and generating with different models;
        # only store n sig and n bkg pulls
        pull = np.empty(2)
        for i in range(2):
            pull[i] = (true_params[i] - fit_params[i]) / fit_errs[i]

    else:
        pull = (true_params - fit_params) / fit_errs

    return pull


def _pull_study(
    n_experiments: int,
    n_evts: Tuple[int, int],
    sign: str,
    time_bin: int,
    out_list: list,
    out_dict: dict,
    sqrt_gen: bool,
    sqrt_fit: bool,
) -> None:
    """
    Return arrays of pulls for the 9 fit parameters

    return value appended to out_list; (9xN) shape array of pulls

    """
    n_sig, n_bkg = n_evts

    rng = np.random.default_rng(seed=(os.getpid() * int(time.time())) % 123456789)

    # 10 params for the sqrt bkg fit; 11 for the phenomenological fit
    n_params = 10 if sqrt_fit else 11

    return_vals = tuple([] for _ in range(n_params))
    for _ in tqdm(range(n_experiments)):
        pulls = _pull(
            rng,
            n_sig,
            n_bkg,
            sign,
            time_bin,
            out_dict,
            sqrt_gen=sqrt_gen,
            sqrt_fit=sqrt_fit,
        )

        if sqrt_gen is sqrt_fit:
            # We're fitting to and generating the same model
            for lst, val in zip(return_vals, pulls):
                lst.append(val)
        else:
            # We're fitting to and generating with different models
            # Only look at the pulls of n_sig and n_bkg
            return_vals[0].append(pulls[0])
            return_vals[1].append(pulls[1])
            for lst in return_vals[2:]:
                lst.append(np.nan)

    out_list.append(np.array(return_vals))


def _plot_pulls(
    fig,
    ax,
    pulls: Tuple,
    sqrt_gen: bool,
    sqrt_fit: bool,
) -> None:
    """
    Plot pulls

    """
    labels = (
        "n signal",
        "n bkg",
        "centre",
        "width L",
        "width R",
        r"$\alpha_L$",
        r"$\alpha_R$",
        r"$\beta$",
        "a",
        "b",
    )
    if not sqrt_fit:
        labels = tuple(list(labels) + ["c"])

    if sqrt_gen != sqrt_fit:
        # Fitting to and generating different models;
        # only plot n sig and n bkg pulls
        for i in range(2):
            axis = ax.ravel()[i]
            pull = pulls[i]

            axis.hist(pull, label=f"{np.mean(pull):.4f}+-{np.std(pull):.4f}", bins=20)
            axis.set_title(labels[i])
            axis.legend()

    else:
        for a, p, l in zip(ax.ravel(), pulls, labels):
            a.hist(p, label=f"{np.mean(p):.4f}+-{np.std(p):.4f}", bins=20)
            a.set_title(l)
            a.legend()

    fig.suptitle(
        f"Generated: {'sqrt' if sqrt_gen else 'alt'} model; Fit: {'sqrt' if sqrt_fit else 'alt'} model"
    )


def _do_pull_study(args: argparse.Namespace):
    """
    Do the pull study

    """
    sign, time_bin = "cf", 5
    # NB these are the total number generated BEFORE we do the accept reject
    # The bkg acc-rej is MUCH more efficient than the signal!
    n_sig, n_bkg = 50000, 50000

    manager = Manager()
    out_list = manager.list()
    out_dict = manager.dict()

    # For plotting pulls
    fig, ax = plt.subplots(4, 3, figsize=(12, 9))

    n_procs = 7
    n_experiments = 75
    procs = [
        Process(
            target=_pull_study,
            args=(
                n_experiments,
                (n_sig, n_bkg),
                sign,
                time_bin,
                out_list,
                out_dict,
                args.sqrt_gen,
                args.sqrt_fit,
            ),
        )
        for _ in range(n_procs)
    ]

    for p in procs:
        p.start()

        # So the processes start at a different time and have different seeds
        time.sleep(1)

    for p in procs:
        p.join()

    pulls = np.concatenate(out_list, axis=1)

    gen_str = "_sqrt_gen" if args.sqrt_gen else ""
    fit_str = "_sqrt_fit" if args.sqrt_fit else ""

    _plot_pulls(
        fig,
        ax,
        pulls,
        args.sqrt_gen,
        args.sqrt_fit,
    )

    _fit(
        out_dict["fit_params"],
        out_dict["bkg_params"],
        out_dict["combined"],
        ax.ravel()[-1],
        args.sqrt_fit,
    )
    ax.ravel()[-1].set_title("Example fit")
    fig.tight_layout()

    path = f"pulls_{sign}_{time_bin=}_{n_sig=}_{n_bkg=}_{n_procs=}_{n_experiments=}{gen_str}{fit_str}.png"
    print(f"plotting {path}")
    fig.savefig(path)

    plt.show()
    plt.close(fig)


def main():
    """
    do a pull study

    """
    parser = argparse.ArgumentParser(description="Toy pull study for mass fitter")

    parser.add_argument(
        "--sqrt_gen",
        help="Generate points using the sqrt bkg model",
        action="store_true",
    )
    parser.add_argument(
        "--sqrt_fit", help="Fit using the sqrt bkg model", action="store_true"
    )

    _do_pull_study(parser.parse_args())


if __name__ == "__main__":
    main()
