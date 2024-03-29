"""
Functions for getting the dataframes once they've been dumped

"""
import os
import sys
import glob
import pickle
import pathlib
from typing import Generator, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
from lib_cuts.definitions import Classifier as CutClassifier
from . import definitions, training_vars


def ampgen(sign: str) -> pd.DataFrame:
    """
    Get the AmpGen dataframe

    Sign should be cf or dcs

    """
    try:
        with open(definitions.ampgen_dump(sign), "rb") as df_f:
            return pickle.load(df_f)

    except FileNotFoundError as err:
        print("=" * 79)
        print(f"create the {sign} ampgen dump by running `create_ampgen.py`")
        print("=" * 79)

        raise err


def particle_gun(
    year: str, sign: str, magnetisation: str, show_progress: bool = False
) -> pd.DataFrame:
    """
    Get the particle gun dataframes, concatenate them and return

    :param year: "2018" or "2017". they're the same
    :param sign: "cf" or "dcs"
    :param magnetisation: "magdown" or "magup"
    :param show_progress: whether to display a progress bar

    """
    if year == "2017":
        year = "2018"

    dfs = []
    paths = glob.glob(str(definitions.pgun_dir(year, sign, magnetisation) / "*.pkl"))

    progress_fcn = tqdm if show_progress else lambda x: x

    for path in progress_fcn(paths):
        try:
            with open(path, "rb") as df_f:
                dfs.append(pickle.load(df_f))

        except FileNotFoundError as err:
            print("=" * 79)
            print(f"create the {sign} particle gun dump by running `create_pgun.py`")
            print("=" * 79)

            raise err

    return pd.concat(dfs)


def pgun_n_gen_file(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Text file holding information about how many evts
    were generated for each particle gun dataframe

    :param year: year
    :param sign: "cf" or "dcs"
    :param magnetisation: mag dir
    :returns: path object

    """
    if year == "2017":
        year = "2018"

    return definitions.pgun_dir(year, sign, magnetisation) / "n_gen.txt"


def pgun_n_generated(year: str, sign: str, magnetisation: str) -> int:
    """
    Number of events generated when creating particle gun dataframes

    Run `create_pgun.py` first for this to work!

    :param sign: "cf" or "dcs"
    :returns: total number of events that were generated when reconstructing the events
              in the pgun dataframes

    """
    n_tot = 0
    with open(str(pgun_n_gen_file(year, sign, magnetisation)), "r") as gen_f:
        for line in gen_f:
            n_tot += int(line.strip())

    return n_tot


def absolute_efficiency(year: str, sign: str, magnetisation: str) -> int:
    """
    Get the absolute efficiency from particle gun dataframes

    """
    return len(particle_gun(year, sign, magnetisation)) / pgun_n_generated(
        year, sign, magnetisation
    )


def abs_eff_err(year: str, sign: str, magnetisation: str) -> int:
    """
    Get the statistical error on absolute efficiency from
    particle gun dataframes

    """
    num = len(particle_gun(year, sign, magnetisation))
    denom = pgun_n_generated(year, sign, magnetisation)

    return num * np.sqrt(1 / num + 1 / denom) / denom


def false_sign(show_progress: bool = False) -> pd.DataFrame:
    """
    Get the false sign particle gun dataframes, concatenate them and return

    :param show_progress: whether to display a progress bar

    """
    dfs = []
    paths = glob.glob(str(definitions.FALSE_SIGN_DIR / "*"))

    progress_fcn = tqdm if show_progress else lambda x: x

    for path in progress_fcn(paths):
        try:
            with open(path, "rb") as df_f:
                dfs.append(pickle.load(df_f))

        except FileNotFoundError as err:
            print("=" * 79)
            print("create the false sign dumps by running `create_false_sign.py`")
            print("=" * 79)

            raise err

    return pd.concat(dfs)


def mc(year: str, sign: str, magnetisation: str) -> pd.DataFrame:
    """
    Get a MC dataframe

    Sign should be cf or dcs

    """
    assert year in {"2017", "2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magup", "magdown"}

    try:
        with open(definitions.mc_dump(year, sign, magnetisation), "rb") as df_f:
            return pickle.load(df_f)

    except FileNotFoundError as err:
        print("=" * 79)
        print(f"create the {sign} MC dump by running `create_mc.py`")
        print("=" * 79)

        raise err


def uppermass(
    year: str, sign: str, magnetisation: str
) -> Generator[pd.DataFrame, None, None]:
    """
    Get the upper mass sideband dataframes;
    they might be quite big so this is a generator

    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"

    """
    paths = glob.glob(str(definitions.uppermass_dir(year, sign, magnetisation) / "*"))
    for path in paths:
        with open(path, "rb") as df_f:
            yield pickle.load(df_f)


def data(
    year: str, sign: str, magnetisation: str
) -> Generator[pd.DataFrame, None, None]:
    """
    Get the real data dataframes; they might be quite big so this is a generator

    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"

    """
    paths = glob.glob(str(definitions.data_dir(year, sign, magnetisation) / "*"))
    for path in paths:
        if not os.path.isdir(path):
            with open(path, "rb") as df_f:
                yield pickle.load(df_f)


def cut_data(year: str, sign: str, magnetisation: str, clf: CutClassifier):
    """
    Get the real data dataframes, applying the signal cut using the provided classifier
    Dataframes might be quite big so this is a generator

    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"

    """
    training_labels = list(training_vars.training_var_names())
    threshhold = 0.185
    for dataframe in data(year, sign, magnetisation):
        predicted_signal = (
            clf.predict_proba(dataframe[training_labels])[:, 1] > threshhold
        )
        yield dataframe[predicted_signal]


def binned_generator(
    generator: Generator[pd.DataFrame, None, None], phsp_bin: int
) -> Generator[pd.DataFrame, None, None]:
    """
    Given a generator of dataframes, returns another generator of dataframes selecting
    only events in the desired phase space bin

    "phsp bin" column must exist in the dataframe

    :param generator: generator of dataframes
    :param phsp_bin: phsp bin index, 0/1/2/3 or None
                     if None, this fcn returns all of them (but not the Ks veto)

    """
    if phsp_bin is None:
        for dataframe in generator:
            yield dataframe[np.isin(dataframe["phsp bin"], (0, 1, 2, 3))]

    for dataframe in generator:
        yield dataframe[dataframe["phsp bin"] == phsp_bin]


def time_binned_generator(
    dataframes: Iterable[pd.DataFrame], low_t: float, high_t: float
) -> Iterable[pd.DataFrame]:
    """
    Generator of dataframes, selecting events between the provided times

    :parameter dataframes: Iterable (can be a generator) of dataframes
    :parameter low_t: low time for bin (exclusive)
    :parameter high_t: low time for bin (exclusive)

    :returns: generator of dataframes, sliced such that only times we want are kept

    """
    assert low_t < high_t

    for dataframe in dataframes:
        times = dataframe["time"]

        keep = (low_t < times) & (times < high_t)

        yield dataframe[keep]
