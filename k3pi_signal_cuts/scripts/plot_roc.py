"""
Plot ROC curve for our classifier

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import util
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def _best_threshhold(fpr, tpr, threshhold):
    """
    Find the fpr, tpr and threshhold closest to (0, 1)

    """
    points = np.column_stack((fpr, tpr))
    distances = np.linalg.norm(points - (0, 1), axis=1)

    min_index = np.argmin(distances)

    return fpr[min_index], tpr[min_index], threshhold[min_index]


def main():
    """
    ROC curve

    """
    # Read dataframes of stuff
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Lets also undersample so we get the same amount of signal/bkg that we expect to see
    # in the data
    sig_frac = 0.0969
    keep_frac = util.weight(
        np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df)))), sig_frac
    )
    sig_keep = np.random.default_rng().random(len(sig_df)) < keep_frac

    sig_df = sig_df[sig_keep]

    combined_df = pd.concat((sig_df, bkg_df))
    combined_y = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    training_labels = list(training_vars.training_var_names())
    predicted_probs = get_clf(year, sign, magnetisation).predict_proba(
        combined_df[training_labels]
    )[:, 1]
    fpr, tpr, threshholds = roc_curve(combined_y, predicted_probs)
    score = roc_auc_score(combined_y, predicted_probs)

    # Find the threshhold closest to the top left corner
    best_fpr, best_tpr, best_threshhold = _best_threshhold(fpr, tpr, threshholds)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"score={score:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    # Plot the best threshhold
    ax.plot([best_fpr], [best_tpr], "ro", label=f"threshhold: {best_threshhold:.3f}")

    ax.legend()
    fig.tight_layout()

    plt.savefig("roc.png")

    plt.show()


if __name__ == "__main__":
    main()
