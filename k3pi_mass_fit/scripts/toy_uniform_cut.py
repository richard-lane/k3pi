"""
Study showing the effect of classifier cuts on a toy dataset

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
from hep_ml.uboost import uBoostBDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "libFit"))

import plots


def fill_dataframe(df: pd.DataFrame, seed: int = 0):
    """
    Fill dataframe with random numbers

    """
    # Signal, background should have slightly different distributions
    sig_correlation = np.array(
        [
            [1.0, 0.8, 0.7, 0.6],
            [0.8, 1.2, 0.5, 0.5],
            [0.7, 0.5, 1.4, 0.5],
            [0.6, 0.5, 0.5, 0.8],
        ]
    )
    bkg_correlation = np.array(
        [
            [0.9, 0.6, 0.5, 0.8],
            [0.6, 1.0, 0.5, 0.2],
            [0.5, 0.5, 1.0, 0.6],
            [0.8, 0.2, 0.6, 1.1],
        ]
    )
    sig_mean = np.array([1.0, 2.0, 3.0, 4.0])
    bkg_mean = np.array([0.0, 1.5, 3.5, 4.5])

    gen = np.random.default_rng(seed=seed)

    n_sig, n_bkg = 28000, 20000
    signal = gen.multivariate_normal(sig_mean, sig_correlation, n_sig).T
    bkg = gen.multivariate_normal(bkg_mean, bkg_correlation, n_bkg).T

    labels = ("X", "Y", "Z", "t")
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    kw = {"histtype": "step", "bins": np.linspace(-5, 10, 100)}
    for a, s, b, l in zip(ax.ravel(), signal, bkg, labels):
        a.hist(s, **kw, label="signal")
        a.hist(b, **kw, label="bkg")
        a.set_title(l)
    ax[0, 0].legend()
    fig.savefig("projections.png")

    for s, b, l in zip(signal, bkg, labels):
        df[l] = np.concatenate((s, b))
    df["label"] = np.concatenate(
        (np.ones(n_sig, dtype=np.int8), np.zeros(n_bkg, dtype=np.int8))
    )


def _efficiency_plot(t_bins, t, pred_sig, pred_bkg, true_sig, true_bkg, title, path):
    fig, ax = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    plots.plot_ratio(
        ax[0],
        t[pred_sig],
        t[pred_bkg],
        t_bins,
        "predicted",
    )
    plots.plot_ratio(ax[0], t[true_sig], t[true_bkg], t_bins, "true")
    ax[0].set_ylabel("cut efficiency")

    hist_kw = {"histtype": "step", "bins": np.linspace(t_bins[0], t_bins[-1], 100)}
    ax[1].hist(t[true_sig], **hist_kw, label="Signal")
    ax[1].hist(t[true_bkg], **hist_kw, label="Bkg")

    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("t")
    ax[0].set_ylim(0.0, 3.0)
    fig.suptitle(title)
    fig.tight_layout()

    print(f"plotting {path}")
    fig.savefig(path)
    plt.close(fig)


def _random_forest_plots(df: pd.DataFrame, seed: int = 0):
    """
    Classify and make plots with the provided data

    """
    # Turn the dataframe into points because that's what we'll train the classifier on
    x = np.column_stack([df[a] for a in ["X", "Y", "Z"]])
    y = np.asarray(df["label"])
    t = np.asarray(df["t"])

    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
        x, y, t, random_state=seed
    )

    clf = RandomForestClassifier(n_estimators=500, max_depth=12, max_features=0.8)
    clf.fit(x_train, y_train)

    train_report = classification_report(y_train, clf.predict(x_train))
    with open("random_forest_train.txt", "w") as f:
        f.write(train_report)

    test_report = classification_report(y_test, clf.predict(x_test))
    with open("random_forest_test.txt", "w") as f:
        f.write(test_report)

    fig, ax = plt.subplots()
    plots.roc(ax, clf.predict_proba(x_train)[:, -1], y_train, "train")
    plots.roc(ax, clf.predict_proba(x_test)[:, -1], y_test, "test")
    ax.legend()
    fig.suptitle("Random Forest ROC")
    fig.savefig("random_forest_roc.png")

    predictions_test = clf.predict(x_test)
    t_bins = np.concatenate(
        ([0.5], np.quantile(t, [0.1 * i for i in range(1, 10)]), [10])
    )
    _efficiency_plot(
        t_bins,
        t_test,
        predictions_test == 1,
        predictions_test == 0,
        y_test == 1,
        y_test == 0,
        "Random Forest Test",
        "random_forest_test.png",
    )
    predictions_train = clf.predict(x_train)
    predictions_train = clf.predict(x_train)
    _efficiency_plot(
        t_bins,
        t_train,
        predictions_train == 1,
        predictions_train == 0,
        y_train == 1,
        y_train == 0,
        "Random Forest Train",
        "random_forest_train.png",
    )
    _efficiency_plot(
        t_bins,
        t_train[y_train == 1],
        predictions_train[y_train == 1] == 1,
        predictions_train[y_train == 1] == 0,
        y_train[y_train == 1] == 1,
        y_train[y_train == 1] == 0,
        "Random Forest Train, signal only",
        "random_forest_train_sig.png",
    )
    _efficiency_plot(
        t_bins,
        t_train[y_train == 0],
        predictions_train[y_train == 0] == 1,
        predictions_train[y_train == 0] == 0,
        y_train[y_train == 0] == 1,
        y_train[y_train == 0] == 0,
        "Random Forest Train, bkg only",
        "random_forest_train_bkg.png",
    )
    _efficiency_plot(
        t_bins,
        t_test[y_test == 1],
        predictions_test[y_test == 1] == 1,
        predictions_test[y_test == 1] == 0,
        y_test[y_test == 1] == 1,
        y_test[y_test == 1] == 0,
        "Random Forest Test, signal only",
        "random_forest_test_sig.png",
    )
    _efficiency_plot(
        t_bins,
        t_test[y_test == 0],
        predictions_test[y_test == 0] == 1,
        predictions_test[y_test == 0] == 0,
        y_test[y_test == 0] == 1,
        y_test[y_test == 0] == 0,
        "Random Forest Train, bkg only",
        "random_forest_test.png",
    )


def _uniform_clf_plots(df: pd.DataFrame, seed: int = 0, uniforming_rate: float = 1.0):
    """
    Classify and make plots with the provided data

    """
    suffix = str(uniforming_rate).replace(".", "_")
    df_train, df_test = train_test_split(df, random_state=seed)

    clf = RandomForestClassifier(n_estimators=150, max_depth=12, max_features=0.5)
    base_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=750)
    clf = uBoostBDT(
        uniform_features=["t"],  # Uniform efficiency in t
        uniform_label=1,  # Uniformity in signal (label 1)
        train_features=["X", "Y", "Z"],  # Train on the other 3 variables
        base_estimator=base_clf,
        target_efficiency=0.5,
        uniforming_rate=uniforming_rate,
        n_estimators=100,
    )
    clf.fit(df_train, df_train["label"])

    train_report = classification_report(df_train["label"], clf.predict(df_train))
    with open(f"uboost_{suffix}_train_report.txt", "w") as f:
        f.write(train_report)

    test_report = classification_report(df_test["label"], clf.predict(df_test))
    with open(f"uboost_{suffix}_test_report.txt", "w") as f:
        f.write(test_report)

    fig, ax = plt.subplots()
    plots.roc(ax, clf.predict_proba(df_train)[:, -1], df_train["label"], "train")
    plots.roc(ax, clf.predict_proba(df_test)[:, -1], df_test["label"], "test")
    ax.legend()
    fig.suptitle("uBoostBDT ROC")
    fig.savefig(f"uboost_roc_{suffix}.png")
    plt.close(fig)

    predictions_test = clf.predict(df_test)
    t_bins = np.concatenate(
        ([0.5], np.quantile(df["t"], [0.1 * i for i in range(1, 10)]), [10])
    )
    _efficiency_plot(
        t_bins,
        df_test["t"],
        predictions_test == 1,
        predictions_test == 0,
        df_test["label"] == 1,
        df_test["label"] == 0,
        "uBoost Test",
        f"uboost_test_{suffix}.png",
    )
    predictions_train = clf.predict(df_train)
    _efficiency_plot(
        t_bins,
        df_train["t"],
        predictions_train == 1,
        predictions_train == 0,
        df_train["label"] == 1,
        df_train["label"] == 0,
        "uBoost Train",
        f"uboost_train_{suffix}.png",
    )
    _efficiency_plot(
        t_bins,
        df_train["t"][df_train["label"] == 1],
        predictions_train[df_train["label"] == 1] == 1,
        predictions_train[df_train["label"] == 1] == 0,
        df_train["label"][df_train["label"] == 1] == 1,
        df_train["label"][df_train["label"] == 1] == 0,
        "uBoost Train, signal only",
        f"uboost_train_sig_{suffix}.png",
    )
    _efficiency_plot(
        t_bins,
        df_train["t"][df_train["label"] == 0],
        predictions_train[df_train["label"] == 0] == 1,
        predictions_train[df_train["label"] == 0] == 0,
        df_train["label"][df_train["label"] == 0] == 1,
        df_train["label"][df_train["label"] == 0] == 0,
        "uBoost Train, bkg only",
        f"uboost_train_bkg_{suffix}.png",
    )
    _efficiency_plot(
        t_bins,
        df_test["t"][df_test["label"] == 0],
        predictions_test[df_test["label"] == 0] == 1,
        predictions_test[df_test["label"] == 0] == 0,
        df_test["label"][df_test["label"] == 0] == 1,
        df_test["label"][df_test["label"] == 0] == 0,
        "uBoost Test, bkg only",
        f"uboost_test_bkg_{suffix}.png",
    )
    _efficiency_plot(
        t_bins,
        df_test["t"][df_test["label"] == 1],
        predictions_test[df_test["label"] == 1] == 1,
        predictions_test[df_test["label"] == 1] == 0,
        df_test["label"][df_test["label"] == 1] == 1,
        df_test["label"][df_test["label"] == 1] == 0,
        "uBoost Test, signal only",
        f"uboost_test_sig_{suffix}.png",
    )


def main():
    matplotlib.use("agg")
    # Make some made up dataframes
    df = pd.DataFrame()

    # Fill them with gaussians
    fill_dataframe(df, seed=0)

    procs = [Process(target=_random_forest_plots, args=(df,), kwargs={"seed": 0})]
    for r in (0.0, 1.0, 2.0, 5.0, 10.0):
        procs.append(
            Process(
                target=_uniform_clf_plots,
                args=(df,),
                kwargs={"seed": 0, "uniforming_rate": r},
            )
        )
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
