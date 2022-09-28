"""
Useful definitions and things

"""
import pathlib
from typing import List
from sklearn.ensemble import GradientBoostingClassifier as Classifier


def classifier_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Where the classifier lives

    """
    assert year in {"2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magdown"}

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "classifiers"
        / f"{year}_{sign}_{magnetisation}.pkl"
    )
