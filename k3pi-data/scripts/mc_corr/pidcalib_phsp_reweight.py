"""
Reweight data using the weights from the
PIDcalib histograms

"""
import sys
import pickle
import pathlib
import argparse
from typing import Tuple, Iterable
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_data import corrections, get, util


def main(*, year: str, magnetisation: str, sign: str):
    """
    Read MC, find the right weights, plot histograms before/after reweighting

    """
    # Get real data generators
    n_dfs = 50
    dfs = islice(get.data(year, sign, magnetisation), n_dfs)

    # Find weights using the PID histogram
    pi_hist, k_hist = _get_hists()
    eta_s = []
    p_s = []
    for eta, p in eta_p:
        print(eta.shape, p.shape)
    eta_s = np.concatenate(eta_s)
    p_s = np.concatenate(p_s)

    wt = corrections.pid_weights(eta_s, p_s, pi_hist, k_hist)
    print(wt)

    # plot
    fig, ax = plt.subplots(4, 2, figsize=(14, 7))
    kw = {"density": False, "histtype": "step"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot PIDcalib reweighting in phase space variables"
    )
    parser.add_argument(
        "year", type=str, choices={"2016", "2017", "2018"}, help="Data taking year"
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown", "magup"},
        help="Magnetisation Direction",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="data type",
    )
    main(**vars(parser.parse_args()))
