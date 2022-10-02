"""
Likelihoods from charm threshhold experiments (CLEO and BES-III)

"""
import os
import sys
import ctypes
import pathlib


def _cleo_path() -> pathlib.Path:
    """
    Where the CLEO library lives

    """
    return pathlib.Path(__file__).resolve().parents[0] / "cleo.so"


def _cleo_exists() -> bool:
    """
    Whether the CLEO library has been created yet

    """
    return os.path.exists(_cleo_path())


def cleo_fcn():
    """
    If we are evaluating the CLEO likelihood lots of times in a tight loop
    it may be too slow to open the DLL every time we want to call the function

    If this is the case, you might want to look it up beforehand

    """
    cleo_lib = ctypes.cdll.LoadLibrary(_cleo_path())
    fcn = getattr(cleo_lib, "cleoLikelihood")

    # Tell python about the argument/return types of this function
    fcn.argtypes = [
        ctypes.c_short,
        *[ctypes.c_double] * 4,
    ]
    fcn.restype = ctypes.c_double

    return fcn


def cleo(
    bin_number: int, z_re: float, z_im: float, y: float, r: float, fcn=None
) -> float:
    """
    CLEO log-likelihood, not multiplied by -2

    :param bin_number: phase space bin number, 0->3
    :param z_re: real part of interference parameter
    :param z_im: imaginary part of interference parameter
    :param y: mixing parameter y
    :param r: DCS/CF amplitude ratio
    :param fcn: CLEO likelihood function taken from the library.
                If not provided, will be looked up

    :returns: the log likelihood from CLEO
    :raises AssertionError: if bad bin number passed in
    :raises FileNotFoundError: if the CLEO shared lib hasn't been created yet

    """
    assert bin_number in {0, 1, 2, 3}
    if not _cleo_exists():
        print("Create the CLEO library by running the build script", file=sys.stderr)
        raise FileNotFoundError(_cleo_path())

    if not fcn:
        fcn = cleo_fcn()

    return fcn(bin_number, z_re, z_im, y, r)
