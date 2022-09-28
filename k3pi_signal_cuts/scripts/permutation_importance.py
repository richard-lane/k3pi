"""
Plot permutation-based feature importance for the classifier

"""
import sys
import pathlib
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_data import training_vars, get
from lib_cuts.get import classifier as get_clf
from lib_cuts import metrics, util


def _significance(n_sig: int, n_bkg: int, scale_factor: float) -> float:
    """
    Find the signal significance given predictions for signal/background

    """
    return metrics.signal_significance(scale_factor * n_sig, scale_factor * n_bkg)


def _sig(clf, sig_df: pd.DataFrame, bkg_df: pd.DataFrame) -> float:
    """
    Find the signal significance

    """
    total_expected = 1697700  # Total DCS magdown expected in signal region
    scale_factor = total_expected / (len(sig_df) + len(bkg_df))

    training_labels = list(training_vars.training_var_names())

    threshold = 0.198
    # Perform cuts to the dataframes
    n_sig = np.sum(clf.predict_proba(sig_df[training_labels])[:, 1] > threshold)
    n_bkg = np.sum(clf.predict_proba(bkg_df[training_labels])[:, 1] > threshold)

    # Find the signal significance we're left with
    return _significance(n_sig, n_bkg, scale_factor)


def _roc_auc(clf, sig_df: pd.DataFrame, bkg_df: pd.DataFrame) -> float:
    """
    Find the ROC AUC

    """
    true_labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    var_names = list(training_vars.training_var_names())
    probs = clf.predict_proba(pd.concat((sig_df[var_names], bkg_df[var_names])))[:, 1]

    return roc_auc_score(true_labels, probs)


def _shuffle(
    gen: np.random.Generator, dataframe: pd.DataFrame, label: str
) -> pd.DataFrame:
    """
    Randomly shuffle the indicated column in the dataframe

    Returns a copy

    """
    copy = dataframe.copy()

    copy[label] = gen.permutation(copy[label].values)

    return copy


def _violplot(importances: List[List[float]], path: str) -> Tuple[plt.Figure, plt.Axes]:
    """
    Make a violin plot of permutation importances

    """
    fig, ax = plt.subplots()

    ax.violinplot(importances)

    labels = training_vars.training_var_names()
    ax.set_xticks([i + 1 for i, _ in enumerate(labels)])
    ax.set_xticklabels(labels, rotation=90)

    ax.plot(np.linspace(*ax.get_xlim()), np.zeros(50), "k--")

    fig.suptitle("Feature Importance")
    ax.set_ylabel(r"$\Delta x$")

    fig.tight_layout()
    fig.savefig(path)

    return fig, ax


def _delta_significance(
    gen: np.random.Generator,
    clf,
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    label: str,
):
    """
    Find the change in signal significance if we randomly shuffle
    the values in the column specified

    """
    all_var_sig = _sig(clf, sig_df, bkg_df)
    return all_var_sig - _sig(
        clf, _shuffle(gen, sig_df, label), _shuffle(gen, bkg_df, label)
    )


def _delta_roc(
    gen: np.random.Generator,
    clf,
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    label: str,
):
    """
    Find the change in ROC AUC if we randomly shuffle
    the values in the column specified

    """
    all_var_roc = _roc_auc(clf, sig_df, bkg_df)
    return all_var_roc - _roc_auc(
        clf, _shuffle(gen, sig_df, label), _shuffle(gen, bkg_df, label)
    )


def _importance(
    gen: np.random.Generator,
    clf,
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    objective_fcn,
) -> List[List[float]]:
    """
    Find the permutation importances using the given objective function

    """
    n_trials = 2
    labels = training_vars.training_var_names()
    importances = []
    with tqdm(total=n_trials * len(labels)) as pbar:
        for label in labels:
            sigs = []
            for _ in range(n_trials):
                sigs.append(objective_fcn(gen, clf, sig_df, bkg_df, label))
                pbar.update(1)

            importances.append(sigs)

    return importances


def main():
    """
    Plot feature importances by finding the decrease in signal significance
    as we randomly shuffle each variable

    """
    # Get the testing data
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
    print(f"keeping {np.sum(sig_keep)} of {len(sig_keep)}")

    print(
        f"sig frac {np.sum(sig_keep):,} / {np.sum(sig_keep) + len(bkg_df):,}"
        f"= {100 * np.sum(sig_keep) / (np.sum(sig_keep) + len(bkg_df)):.4f}%"
    )

    # Get the classifier
    clf = get_clf(year, sign, magnetisation)

    # Find the signal significance if we shuffle one of the columns
    gen = np.random.default_rng(seed=0)
    sig_importances = _importance(gen, clf, sig_df, bkg_df, _delta_significance)
    roc_importances = _importance(gen, clf, sig_df, bkg_df, _delta_roc)

    _violplot(sig_importances, "significance_importances.png")
    plt.show()

    _violplot(roc_importances, "roc_importances.png")
    plt.show()


if __name__ == "__main__":
    main()
