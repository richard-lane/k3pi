"""
Sanity test for the classifier

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..classifier import Classifier


def test_scaling():
    """
    Check that even if we train on an imbalanced dataset, the predicted probabilities have a mean of 0.5

    """
    gen = np.random.default_rng()
    n_sig = 10000
    n_bkg = 10000 * 4

    sig = gen.exponential(scale=2, size=n_sig).reshape(-1, 1)
    bkg = gen.exponential(scale=4, size=n_bkg).reshape(-1, 1)

    points = np.concatenate((sig, bkg))
    labels = np.concatenate((np.ones(len(sig)), np.zeros(len(bkg))))

    clf = Classifier(n_sig, n_bkg, RandomForestClassifier())
    clf.clf.fit(points, labels)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    bins = np.linspace(0, 10, 100)
    ax[0].hist(sig, bins=bins, histtype="step")
    ax[0].hist(bkg, bins=bins, histtype="step")

    bins = np.linspace(0, 1, 100)
    print(clf.predict_proba(sig))
    ax[1].hist(clf.predict_proba(sig)[:, 0], histtype="step", bins=bins)
    ax[1].hist(clf.predict_proba(bkg)[:, 0], histtype="step", bins=bins)

    plt.show()
