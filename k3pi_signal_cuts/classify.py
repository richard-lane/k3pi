"""
TODO delete
Train a classifier to separate signal and background

You should have already downloaded the data files and created pickle dumps

"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from lib_cuts import util
from lib_cuts import read_data
from lib_cuts import definitions


def _train_test_dfs():
    """ Join signal + bkg dataframes but split by train/test """
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    combined_df = pd.concat((sig_df, bkg_df))

    # 1 for signal, 0 for background
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    train_mask = combined_df["train"]
    return (
        combined_df[train_mask],
        combined_df[~train_mask],
        labels[train_mask],
        labels[~train_mask],
    )


def main():
    """
    Create the classifier, print training scores

    """
    train_df, test_df, train_label, test_label = _train_test_dfs()

    clf = RandomForestClassifier()

    # We only want to train on a subset of the columns in the dataframe
    # i.e. we don't want to train based on the D mass, Delta M or the train/test label
    training_labels = list(read_data.training_var_names())
    clf.fit(train_df[training_labels], train_label)

    print(classification_report(train_label, clf.predict(train_df[training_labels])))
    print(classification_report(test_label, clf.predict(test_df[training_labels])))

    with open(definitions.CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
