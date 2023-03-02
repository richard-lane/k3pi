"""
Get a classifier

"""
import sys
import pickle
import pathlib
from typing import Iterable, Tuple
from functools import partial
import pandas as pd
from . import definitions

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import training_vars, get
from lib_data.util import no_op


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
    training_labels = list(training_vars.training_var_names())
    predicted_signal = clf.predict_proba(dataframe[training_labels])[:, 1] > threshold

    return dataframe[predicted_signal]


def cut_dfs(
    dataframes: Iterable[pd.DataFrame],
    clf: definitions.Classifier = None,
    threshold: float = definitions.THRESHOLD,
    *,
    perform_cut: bool = True,
) -> Iterable[pd.DataFrame]:
    """
    Apply the BDT cut to each dataframe

    :param dataframes: iterable (may be a generator)
    :param clf: classifier for performing BDT cut
    :param threshold: classifier threshold
    :param perform_cut: if false, this function doesn't
                        modify the dataframes. Useful if we don't
                        want to perform the BDT cut, but still want
                        to have this function somewhere in our analysis
                        pipeline.

    :returns: generator of dataframes

    """
    func = (
        partial(signal_cut_df, clf=clf, threshold=threshold) if perform_cut else no_op
    )

    for dataframe in dataframes:
        yield func(dataframe)


def time_cut_dfs(
    year: str,
    magnetisation: str,
    sign: str,
    bdt_cut: bool,
    time_range: Tuple[float, float],
) -> Iterable[pd.DataFrame]:
    """
    Get a generator of dataframes from the right dump,
    cut according to the time limits

    """
    dfs = get.data(year, sign, magnetisation)

    # Get the BDT if we need to
    bdt_clf = classifier(year, "dcs", magnetisation) if bdt_cut else None

    # This fcn doesn't actually modify the dataframes if we
    # don't provide a classifier
    dataframes = cut_dfs(dfs, bdt_clf, perform_cut=bdt_cut)

    for dataframe in dataframes:
        times = dataframe["time"]
        keep = (time_range[0] < times) & (times < time_range[1])

        yield dataframe[keep]
