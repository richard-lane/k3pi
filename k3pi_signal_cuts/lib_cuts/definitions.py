"""
Useful definitions and things

"""
import pathlib
from sklearn.ensemble import GradientBoostingClassifier as Classifier

# Expected absolute number in the signal region, roughly
EXPECTED_N_SIG_SIG_REGION = 2_500
EXPECTED_N_BKG_SIG_REGION = 12_000
EXPECTED_N_TOT_SIG_REGION = EXPECTED_N_SIG_SIG_REGION + EXPECTED_N_BKG_SIG_REGION

# Expected signal fraction in the signal region, roughly
EXPECTED_SIG_FRAC = EXPECTED_N_SIG_SIG_REGION / EXPECTED_N_TOT_SIG_REGION

# Optimal threshhold for the BDT - hopefully
THRESHOLD = 0.28


def classifier_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Where the classifier lives

    """
    assert year in {"2017", "2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magup", "magdown"}

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "classifiers"
        / f"{year}_{sign}_{magnetisation}.pkl"
    )
