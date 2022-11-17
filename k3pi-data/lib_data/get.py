"""
Functions for getting the dataframes once they've been dumped

"""
import sys
import glob
import pickle
import pathlib
from typing import Generator
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "k3pi_signal_cuts"))
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


def particle_gun(sign: str, show_progress: bool = False) -> pd.DataFrame:
    """
    Get the particle gun dataframes, concatenate them and return

    :param sign: "cf" or "dcs"
    :param show_progress: whether to display a progress bar

    """
    dfs = []
    paths = glob.glob(str(definitions.pgun_dir(sign) / "*"))

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
    assert year in {"2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magdown"}

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
