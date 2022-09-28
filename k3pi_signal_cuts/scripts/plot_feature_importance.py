"""
Plot impurity-based feature importance for the classifier

"""
import sys
import pathlib
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts.get import classifier as get_clf
from lib_data import training_vars


def main():
    """
    Plot feature importances naively- using an impurity based method

    """
    # Predict which of these are signal and background using our classifier
    clf = get_clf("2018", "dcs", "magdown")

    fig, ax = plt.subplots(figsize=(10, 5))

    importances = clf.feature_importances_
    ax.bar(range(len(importances)), importances)
    ax.set_xticks(
        range(len(importances)), training_vars.training_var_names(), rotation=45
    )

    fig.suptitle("Feature Importances")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
