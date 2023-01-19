"""
k3pi_mass_fit unit test

"""
import sys
import pathlib
import numpy as np
from scipy.integrate import quad

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))
from libFit import pdfs, definitions
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
    points = definitions.mass_bins(1000)

    integral = stats.integral(points, pdfs.normalised_signal(points, *params))

    assert np.isclose(integral, 1.0)


def test_model_normalised():
    """
    Check that the pdf is normalised correctly

    """
    n_sig = 1000
    n_bkg = 10000

    params = (146, 0.2, 0.2, 0.2, 0.2, 0.002, 0.004, -0.001)
    points = definitions.mass_bins(1000)

    integral = stats.integral(points, pdfs.model(points, n_sig, n_bkg, *params))

    assert np.isclose(integral, n_sig + n_bkg)


def test_alt_bkg_normalised():
    """
    Check that the alternate background PDF is normalised

    """
    a, b = pdfs.domain()
    domain_width = b - a

    # Pretend the estimated bkg is a constant
    estimated_bkg_pdf = lambda x: 1 / domain_width

    assert np.isclose(
        quad(lambda x: pdfs.estimated_bkg(x, estimated_bkg_pdf, 0, 0, 0), a, b)[0], 1.0
    )

    assert np.isclose(
        quad(lambda x: pdfs.estimated_bkg(x, estimated_bkg_pdf, 3, 0, 0), a, b)[0], 1.0
    )

    assert np.isclose(
        quad(lambda x: pdfs.estimated_bkg(x, estimated_bkg_pdf, 3, 2, 0), a, b)[0], 1.0
    )

    assert np.isclose(
        quad(lambda x: pdfs.estimated_bkg(x, estimated_bkg_pdf, 3, 2, 1), a, b)[0], 1.0
    )
