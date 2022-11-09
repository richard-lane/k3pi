"""
Apply the BDT cut to some data

"""
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
from lib_cuts.get import classifier as get_clf
from lib_data import training_vars


def mask(
    dataframe: pd.DataFrame,
    year: str,
    magnetisation: str,
    sign: str,
    threshhold: float = 0.185,
) -> np.ndarray:
    """
    Boolean mask of evnets to keep following the BDT cut

    :param dataframe: dataframe including BDT training vars
    :param year: data taking year, used for finding the right BDT
    :param magnetisation: used for finding the right BDT
    :param sign: used for finding the right BDT
    :param threshhold: predicted probability below which we call things background.
           Defaults to some value that i should probably make consistent somewhere TODO
    :returns: boolean masks of events to keep

    """
    # Open reweighter
    clf = get_clf(year, sign, magnetisation)

    # Evaluate bkg probabilities
    # Don't want to keep evts below threshhold
    labels = list(training_vars.training_var_names())
    predictions = clf.predict_proba(dataframe[labels])[:, 1] > threshhold
    return predictions == 1
