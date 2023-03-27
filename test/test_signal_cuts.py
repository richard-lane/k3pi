"""
Unit tests for stuff in the signal cuts directory

"""
import sys
import pathlib
import pytest
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_signal_cuts"))
from lib_cuts.metrics import signal_significance
from lib_cuts import util


def test_signal_significance():
    """
    Check we get the right signal significance out

    """
    n_sig, n_bkg = 5, 10
    assert signal_significance(n_sig, n_bkg) == 5 / np.sqrt(15)


def test_signal_significance_array():
    """
    Check we get the right signal significances out for ararys

    """
    n_sig, n_bkg = np.array((5, 7)), np.array((10, 15))

    assert np.allclose(
        signal_significance(n_sig, n_bkg), np.array((5 / np.sqrt(15), 7 / np.sqrt(22)))
    )


def test_sig_frac_weights():
    """
    Check we get the right weight out when adjusting
    to the right signal significance

    This isn't really a unit test in the technical sense

    """
    n_sig = 100
    n_bkg = 50
    labels = np.concatenate((np.ones(n_sig), np.zeros(n_bkg)))

    desired_sig_frac = 0.5

    wts = util.weights(labels, desired_sig_frac)

    # number of signal evts
    n_sig_weighted = np.sum(wts[labels == 1.0])
    n_bkg_weighted = np.sum(wts[labels == 0.0])

    assert n_sig_weighted / (n_sig_weighted + n_bkg_weighted) == desired_sig_frac


def test_sig_frac_overflow():
    """
    Check we get the right error when the weights are > 1

    Raise an error in this case because this means we are looking to
    increase the amount of signal via our weighting, which is probably
    wrong since our signal cuts are designed to work on WS data where there is
    much more signal than background.

    It's possible that you might not want this to raise an error (e.g. if you really
    are training the signal cut classifier on much much more bkg than signal, and
    therefore need to artificially weight the signal evts upwards), but if this is the case
    you're probably using too much background and you can probably just change the code anyway.

    """
    n_sig = 100
    n_bkg = 500
    labels = np.concatenate((np.ones(n_sig), np.zeros(n_bkg)))
    wts = np.ones_like(labels)

    desired_sig_frac = 0.5

    with pytest.raises(AssertionError):
        util.weight(labels, desired_sig_frac, wts)
