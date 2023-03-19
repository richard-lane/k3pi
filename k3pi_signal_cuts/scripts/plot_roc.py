"""
Plot ROC curve for our classifier

"""
import sys
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import util, definitions
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars, d0_mc_corrections


def _dataframe(
    year: str, sign: str, magnetisation: str, *, train: bool
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Get either the training or testing dataframe
    and corresponding ground truth labels

    """
    # Read dataframes of stuff
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    mc_corr_wts = d0_mc_corrections.mc_weights(year, sign, magnetisation)
    mc_corr_wts /= np.mean(mc_corr_wts)

    # We only want the testing data here
    sig_mask = sig_df["train"] if train else ~sig_df["train"]
    bkg_mask = bkg_df["train"] if train else ~bkg_df["train"]

    mc_corr_wts = mc_corr_wts[sig_mask]
    sig_df = sig_df[sig_mask]
    bkg_df = bkg_df[bkg_mask]

    # Lets also undersample so we get the same amount of signal/bkg that we expect to see
    # in the data
    sig_frac = 0.0969
    keep_frac = util.weight(
        np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df)))),
        sig_frac,
        np.concatenate((mc_corr_wts, np.ones(len(bkg_df)))),
    )
    sig_keep = np.random.default_rng().random(len(sig_df)) < keep_frac

    sig_df = sig_df[sig_keep]

    combined_df = pd.concat((sig_df, bkg_df))
    combined_y = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    return combined_df, combined_y


def _best_threshhold(fpr, tpr, threshhold):
    """
    Find the fpr, tpr and threshhold closest to (0, 1)

    """
    points = np.column_stack((fpr, tpr))
    distances = np.linalg.norm(points - (0, 1), axis=1)

    min_index = np.argmin(distances)

    return fpr[min_index], tpr[min_index], threshhold[min_index]


def _plot_roc(
    axis: plt.Axes,
    dataframe: pd.DataFrame,
    ground_truth: np.ndarray,
    classifier: definitions.Classifier,
    label: str,
    **plot_kw,
) -> None:
    """
    Plot a ROC curve on an axis given the data and classifier

    """

    training_labels = list(training_vars.training_var_names())
    predicted_probs = classifier.predict_proba(dataframe[training_labels])[:, 1]
    fpr, tpr, threshholds = roc_curve(ground_truth, predicted_probs)
    score = roc_auc_score(ground_truth, predicted_probs)

    # Find the threshhold closest to the top left corner
    best_fpr, best_tpr, best_threshhold = _best_threshhold(fpr, tpr, threshholds)

    (line,) = axis.plot(fpr, tpr, label=f"{label} AUC={score:.4f}", **plot_kw)

    # Plot the best threshhold
    axis.plot(
        [best_fpr],
        [best_tpr],
        marker="o",
        label=f"threshhold: {best_threshhold:.3f}",
        color=line.get_color(),
    )


def main():
    """
    ROC curve

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"
    classifier = get_clf(year, sign, magnetisation)

    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], "k--")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")

    _plot_roc(
        axis,
        *_dataframe(year, sign, magnetisation, train=False),
        classifier,
        "Test",
        color="k",
    )
    _plot_roc(
        axis,
        *_dataframe(year, sign, magnetisation, train=True),
        classifier,
        "Train",
        color="r",
        linestyle=":",
    )

    axis.legend()
    fig.tight_layout()

    plt.savefig("roc.png")


if __name__ == "__main__":
    main()
