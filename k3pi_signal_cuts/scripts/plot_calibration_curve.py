"""
Plot a calibration curve for the classifier

If we are using a calibrated classifier, hopefully this will be well-calibrated...

In the case of the Random Forest (which I am using for now..) this is unlikely to be calibrated a priori, since
the ensemble of trees is unlikely to predict either 0 or 1 probability because this would require all trees to
predict (e.g.) 0. Thsi is unlikely as there may be noise in the trees (from e.g. training on a subset of features).

This doesn't matter much though, as long as it is roughly calibrated for the values around 0.5 that we might use for
the predict_proba cut threshhold

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util


def main():
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    # Plot calibration using testing data, because that seems right
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    combined_df = pd.concat((sig_df, bkg_df))
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    # Predict which of these are signal and background using our classifier
    clf = util.get_classifier()
    var_names = list(read_data.training_var_names())

    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", sharex=True)

    n_bins = 11
    CalibrationDisplay.from_estimator(
        clf, combined_df[var_names], labels, n_bins=n_bins, ax=ax["A"]
    )
    ax["B"].hist(clf.predict_proba(combined_df[var_names])[:, 1])
    ax["B"].set_ylabel("count")
    ax["B"].set_xlabel(ax["A"].get_xlabel())
    ax["B"].set_yticklabels(ax["B"].get_yticklabels(), rotation=90)
    ax["A"].set_xlabel("")

    plt.show()


if __name__ == "__main__":
    main()
