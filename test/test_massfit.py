"""
k3pi_mass_fit unit test

"""
import sys
import pathlib
import numpy as np
from scipy.integrate import quad

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))
from libFit import pdfs
from lib_data import stats


def test_bkg_normalised():
    """
    Check the background pdf is normalised

    """
    assert np.isclose(
        quad(pdfs.normalised_bkg, *pdfs.domain(), args=pdfs.background_defaults("WS"))[
            0
        ],
        1.0,
    )


def test_signal_normalised():
    """
    Check the signal pdf is normalised

    """
    params = (146, 0.2, 0.2, 0.2, 0.2, 0.002)
    points = np.linspace(*pdfs.domain(), 1000)

    integral = stats.integral(points, pdfs.normalised_signal(points, *params))

    assert np.isclose(integral, 1.0)


def test_model_normalised():
    """
    Check that the pdf is normalised correctly

    """
    n_sig = 1000
    n_bkg = 10000

    params = (146, 0.2, 0.2, 0.2, 0.2, 0.002, 0.004, -0.001)
    points = np.linspace(*pdfs.domain(), 1000)

    integral = stats.integral(points, pdfs.model(points, n_sig, n_bkg, *params))

    assert np.isclose(integral, n_sig + n_bkg)