"""
Get a classifier

"""
import sys
import pickle
import pathlib
from typing import Iterable
import pandas as pd
from . import definitions

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import training_vars


def classifier(year: str, sign: str, magnetisation: str) -> definitions.Classifier:
    """
    The right classifier

    """
    with open(definitions.classifier_path(year, sign, magnetisation), "rb") as f:
        return pickle.load(f)


def signal_cut_df(
    dataframe: pd.DataFrame, clf: definitions.Classifier, threshold: float
) -> pd.DataFrame:
    """
    Apply the BDT cut to the dataframe; return dataframe after the cut

    """
    # Always use the DCS BDT to do cuts
    training_labels = list(training_vars.training_var_names())
    predicted_signal = clf.predict_proba(dataframe[training_labels])[:, 1] > threshold

    return dataframe[predicted_signal]


def cut_dfs(
    dataframes: Iterable[pd.DataFrame],
    clf: definitions.Classifier,
    threshold: float = definitions.THRESHOLD,
) -> Iterable[pd.DataFrame]:
    """
    Apply the BDT cut to each dataframe

    """
    for dataframe in dataframes:
        yield signal_cut_df(dataframe, clf, threshold)
