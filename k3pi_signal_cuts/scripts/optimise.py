"""
Train several classifiers with different parameters,
record the ROC AUC score at each, make some plots
+ cache the hyperparameters + ROC

"""
import os
import sys
import time
import pickle
import pathlib
import argparse
from multiprocessing import Process, Manager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts import definitions, util
from lib_data import get, training_vars, d0_mc_corrections


def _train_test_dfs(year, sign, magnetisation):
    """
    Join signal + bkg dataframes but split by train/test

    Returns also weights for the MC correction

    """
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(list(get.uppermass(year, sign, magnetisation)))

    # Find MC correction wts for the sig df
    # Scale st their average is 1.0
    mc_corr_wts = d0_mc_corrections.mc_weights(year, sign, magnetisation)
    mc_corr_wts /= np.mean(mc_corr_wts)

    combined_df = pd.concat((sig_df, bkg_df))

    combined_wts = np.concatenate((mc_corr_wts, np.ones(len(bkg_df))))

    # 1 for signal, 0 for background
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    train_mask = combined_df["train"]
    return (
        combined_df[train_mask],
        combined_df[~train_mask],
        labels[train_mask],
        labels[~train_mask],
        combined_wts[train_mask],
        combined_wts[~train_mask],
    )


def _run_study(
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    train_wts: np.ndarray,
    test_df: pd.DataFrame,
    test_labels: np.ndarray,
    n_repeats: int,
    out_dict: dict,
):
    """
    Create the classifier, print training scores

    """
    # Create an RNG - one for each process. Seed with PID
    rng = np.random.default_rng(seed=int(time.time_ns() * os.getpid()) % 123456789)

    results = []
    for _ in tqdm(range(n_repeats)):
        # Choose hyperparams
        train_params = {
            "n_estimators": rng.integers(20, 300),
            "learning_rate": rng.random() ** 3,
            "max_depth": rng.integers(2, 13),
            "loss": "exponential",
        }

        # Train a classifier
        # Type is defined in lib_cuts.definitions
        clf = definitions.Classifier(**train_params)

        # We only want to use some of our variables for training
        training_cols = list(training_vars.training_var_names())
        clf.fit(train_df[training_cols], train_labels, train_wts)

        # Find the training ROC AUC score
        train_score = roc_auc_score(train_labels, clf.predict(train_df[training_cols]))

        # Find the testing ROC AUC score
        test_score = roc_auc_score(test_labels, clf.predict(test_df[training_cols]))

        # Make a dict of the training parameters and ROC AUC scores

        # Append to a list of dicts
        results.append(
            {**train_params, "train_roc_auc": train_score, "test_roc_auc": test_score}
        )

    # When we're done store the list of params to the out dict
    out_dict[os.getpid()] = results


def _dict2arrs(scores: dict) -> tuple:
    """
    Get arrays of parameters from the arrays

    """
    n_estimators = []
    learning_rate = []
    max_depth = []
    train_roc = []
    test_roc = []

    for dict_ in scores.values():
        n_estimators.append([d_["n_estimators"] for d_ in dict_])
        learning_rate.append([d_["learning_rate"] for d_ in dict_])
        max_depth.append([d_["max_depth"] for d_ in dict_])
        train_roc.append([d_["train_roc_auc"] for d_ in dict_])
        test_roc.append([d_["test_roc_auc"] for d_ in dict_])

    # Concatenate into arrays
    return (
        np.concatenate(n_estimators),
        np.concatenate(learning_rate),
        np.concatenate(max_depth),
        np.concatenate(train_roc),
        np.concatenate(test_roc),
    )


def _plot_scores(scores: dict) -> None:
    """
    Plot bar/line charts showing the ROC score in testing/training o

    """
    # Get arrays of params from the dict
    n_estimators, learning_rate, max_depth, train_roc, test_roc = _dict2arrs(scores)

    fig, axes = plt.subplots(2, 3, sharey=True)
    axes[0, 0].scatter(n_estimators, train_roc, color="r", label="Train")
    axes[1, 0].scatter(n_estimators, test_roc, color="b", label="Train")
    axes[1, 0].set_xlabel("n_estimators")

    axes[0, 1].scatter(learning_rate, train_roc, color="r")
    axes[1, 1].scatter(learning_rate, test_roc, color="b")
    axes[1, 1].set_xlabel("learning_rate")

    axes[0, 2].scatter(max_depth, train_roc, color="r")
    axes[1, 2].scatter(max_depth, test_roc, color="b")
    axes[1, 2].set_xlabel("max_depth")

    # Find the top 5 params
    n_max = 5
    indices = np.argpartition(test_roc, -n_max)[-n_max:]
    print("Test ROC\tTrain ROC\tn est\tlearn rate\tdepth")
    for index in indices:
        print(
            f"{test_roc[index]:.3f}\t",
            f"{train_roc[index]:.3f}\t",
            n_estimators[index],
            f"{learning_rate[index]:.3f}\t",
            max_depth[index],
            sep="\t",
        )

    fig.tight_layout()
    fig.savefig("bdt_opt.png")


def main(*, n_procs: int, n_repeats: int):
    """
    Spawn lots of processes to run the optimisation study, then bring the results back together

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"

    # Label 1 for signal; 0 for bkg
    (
        train_df,
        test_df,
        train_label,
        test_label,
        train_d0_wts,
        test_d0_wts,
    ) = _train_test_dfs(year, sign, magnetisation)

    # We want to train the classifier on a realistic proportion of signal + background
    # Get this from running `scripts/mass_fit.py`
    # using this number for now
    sig_frac = 0.0852
    train_weights = util.weights(train_label, sig_frac, train_d0_wts)

    # Resample to get a representative ratio of signal to bkg in our testing set
    test_mask = util.resample_mask(
        np.random.default_rng(), test_label, sig_frac, test_d0_wts
    )
    test_df = test_df[test_mask]
    test_label = test_label[test_mask]

    # Run the study, find a dict of pid: list of dicts showing hyperparams
    out_dict = Manager().dict()

    procs = [
        Process(
            target=_run_study,
            args=(
                train_df,
                train_label,
                train_d0_wts * train_weights,
                test_df,
                test_label,
                n_repeats,
                out_dict,
            ),
        )
        for _ in range(n_procs)
    ]

    # Start the processes at different times so they have
    # different seeds
    for p in procs:
        p.start()
        time.sleep(0.01)
    for p in procs:
        p.join()

    # Plot the ROC scores vs hyperparams
    _plot_scores(out_dict)

    # Cache the dict of lists of dicts to disk so i can recover it later if i want
    with open("bdt_opt.pkl", "wb") as pkl_f:
        pickle.dump(dict(out_dict), pkl_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n_procs",
        type=int,
        help="number of processes to spawn",
        nargs="?",
        default=4,
    )
    parser.add_argument(
        "n_repeats",
        type=int,
        help="number of times to repeat training",
        nargs="?",
        default=10,
    )

    main(**vars(parser.parse_args()))
