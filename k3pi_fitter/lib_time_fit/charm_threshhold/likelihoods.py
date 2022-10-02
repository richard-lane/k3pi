"""
Likelihoods from charm threshhold experiments (CLEO and BES-III)

"""
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
    return _cleo_path().exists()


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


def _bes_cov_mat_path() -> pathlib.Path:
    """
    Path to ROOT file containing BES-III covariance matrix

    Must be acquired by the user

    """
    return pathlib.Path(__file__).resolve().parents[0] / "BESIII_CovMat.root"


def _bes_cov_mat_exists() -> bool:
    """
    Whether the BES-III covariance matrix exists

    """
    return _bes_cov_mat_path().exists()


def _bes_lib_path() -> pathlib.Path:
    """
    Path to the BES-III shared library

    """
    return pathlib.Path(__file__).resolve().parents[0] / "bes.so"


def _bes_lib_exists() -> pathlib.Path:
    """
    Whether the BES-III shared library exists

    """
    return _bes_lib_path().exists()


def bes_fcn():
    """
    If we are evaluating the BES chi2 lots of times in a tight loop
    it may be too slow to open the DLL every time we want to call the function

    If this is the case, you might want to look it up beforehand

    """
    bes_lib = ctypes.cdll.LoadLibrary(_bes_lib_path())
    fcn = getattr(bes_lib, "besChi2")

    # Tell python about the argument/return types of this function
    fcn.argtypes = [
        ctypes.c_short,
        *[ctypes.c_double] * 4,
    ]
    fcn.restype = ctypes.c_double

    return fcn


def bes_chi2(
    bin_number: int, z_re: float, z_im: float, x: float, y: float, fcn=None
) -> float:
    """
    BES chi^2

    :param bin_number: phase space bin number, 0->3
    :param z_re: real part of interference parameter
    :param z_im: imaginary part of interference parameter
    :param y: mixing parameter y
    :param x: mixing parameter x
    :param fcn: BES chi2 function taken from the library.
                If not provided, will be looked up

    :returns: the chi2 from BES-III
    :raises AssertionError: if bad bin number passed in
    :raises FileNotFoundError: if the BES shared lib hasn't been created yet
    :raises FileNotFoundError: if the BES covariance matrix ROOT file isn't in the right place

    """
    assert bin_number in {0, 1, 2, 3}
    if not _bes_cov_mat_exists():
        print(
            "Get the BES-III covariance matrix ROOT file from somewhere"
            f"and save it to {_bes_cov_mat_path()}",
            file=sys.stderr,
        )
        raise FileNotFoundError(_bes_cov_mat_path())
    if not _bes_lib_exists():
        print("Create the BES library by running the build script", file=sys.stderr)
        raise FileNotFoundError(_bes_lib_path())

    if not fcn:
        fcn = bes_fcn()

    return fcn(bin_number, z_re, z_im, x, y)


def combined_chi2(
    bin_number: int,
    z_re: float,
    z_im: float,
    x: float,
    y: float,
    r: float,
    cleo_fcn=None,
    bes_fcn=None,
) -> float:
    """
    BES chi^2

    :param bin_number: phase space bin number, 0->3
    :param z_re: real part of interference parameter
    :param z_im: imaginary part of interference parameter
    :param y: mixing parameter y
    :param x: mixing parameter x
    :param r: DCS/CF amplitude ratio
    :param cleo_fcn: CLEO likelihood function taken from the library.
                If not provided, will be looked up
    :param bes_fcn: BES chi2 function taken from the library.
                If not provided, will be looked up

    :returns: the combined chi2 from BES + CLEO
    :raises AssertionError: if bad bin number passed in
    :raises FileNotFoundError: if either of the BES or CLEO libs
                               haven't been created, or if the BES
                               covariance matrix ROOT file isn't present

    """
    return bes_chi2(bin_number, z_re, z_im, x, y, bes_fcn) - 2 * cleo(
        bin_number, z_re, z_im, y, r, cleo_fcn
    )
