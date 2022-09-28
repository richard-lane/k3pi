"""
Functions for evaluating the amplitude models from python
via the C-compatible libraries

"""
import ctypes
import pathlib
import numpy as np

# Average values of the amplitude models across all of phase space
CF_AVG = 0.08043
DCS_AVG = 0.06707


def _dcs_path() -> pathlib.Path:
    """
    Where the DCS library lives

    """
    return pathlib.Path(__file__).resolve().parents[0] / "dcs_wrapper.so"


def _cf_path() -> pathlib.Path:
    """
    Where the CF library lives

    """
    return pathlib.Path(__file__).resolve().parents[0] / "cf_wrapper.so"


def _manual_free(c_arr) -> None:
    """
    Explicitly free memory used by a C-array

    Necessary because we allocate memory when evaluting the
    amplitude models - the python garbage collector doesn't
    know about this so we have to free the memory by hand

    """
    # manual free fcn is in both the dcs and cf libs, so just
    # pick one
    dll = ctypes.cdll.LoadLibrary(str(_dcs_path()))
    free = getattr(dll, "manual_free")

    free.argtypes = [ctypes.POINTER(ctypes.c_double)]
    free.restype = None

    free(c_arr)


def _amplitudes(
    sign: str,
    k: np.ndarray,
    pi1: np.ndarray,
    pi2: np.ndarray,
    pi3: np.ndarray,
    k_charge: int,
) -> np.ndarray:
    """
    Find the amplitudes for the events passed in for the
    D -> K+ pi- pi+ pi- (or conjugate, if k_charge is -1)

    N.B. - arrays in MeV

    :param sign: "dcs" or "cf"
    :param k: shape (4, N) array of kaon momenta (px, py, pz, e)
    :param pi1: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi2: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi3: shape (4, N) array of pion momenta (px, py, pz, e)
    :param k_charge: charge of the kaon, either +1 or -1

    """
    assert sign in {"cf", "dcs"}
    assert k.dtype == np.float64
    assert pi1.dtype == np.float64
    assert pi2.dtype == np.float64
    assert pi3.dtype == np.float64

    # Get fcn from DLL
    path = str(_dcs_path()) if sign == "dcs" else str(_cf_path())

    dll = ctypes.cdll.LoadLibrary(path)
    fcn = getattr(dll, f"{sign}_wrapper_array")

    # Tell python about its argument/return types
    fcn.argtypes = [
        *[ctypes.POINTER(ctypes.c_double)] * 16,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    fcn.restype = ctypes.POINTER(ctypes.c_double)

    # Convert arrays to C pointers
    # We need to create copies of our arrays here - luckily we also need to divide them by 1000 to convert to
    # GeV, which also creates a copy for us so we don't need to call np.copy explicitly
    k_copies = [(a / 1000).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in k]
    pi1_copies = [
        (a / 1000).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in pi1
    ]
    pi2_copies = [
        (a / 1000).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in pi2
    ]
    pi3_copies = [
        (a / 1000).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in pi3
    ]

    n_evts = len(k.T)
    c_arr = fcn(
        *k_copies,
        *pi1_copies,
        *pi2_copies,
        *pi3_copies,
        ctypes.c_size_t(n_evts),
        ctypes.c_int(k_charge),
    )

    np_arr = np.ctypeslib.as_array(
        (ctypes.c_double * (n_evts * 2)).from_address(ctypes.addressof(c_arr.contents))
    )
    # Copy so that we don't need the original array any more - can free it
    re, im = np.copy(np_arr[::2]), np.copy(np_arr[1::2])

    _manual_free(c_arr)

    amp = re + im * 1.0j

    return amp


def dcs_amplitudes(k, pi1, pi2, pi3, k_charge):
    """
    Find the DCS amplitudes for the events passed in for the
    D -> K+ pi- pi+ pi- (or conjugate, if k_charge is -1)

    NB: arrays should be in MeV; if you have an array in GeV (e.g.
    generated directly by AmpGen), you should scale these yourself
    by multiplying by 1000

    NB: note also that this amplitude is not scaled; to find
    DCS amplitudes that have a relative strong phase of 0 wrt the
    CF amplitudes and that also have the right scaling you should
    scale and rotate these amplitudes by `definitions.DCS_OFFSET`.

    :param k: shape (4, N) array of kaon momenta (px, py, pz, e)
    :param pi1: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi2: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi3: shape (4, N) array of pion momenta (px, py, pz, e)
    :param k_charge: charge of the kaon, either +1 or -1

    """
    return _amplitudes("dcs", k, pi1, pi2, pi3, k_charge)


def cf_amplitudes(k, pi1, pi2, pi3, k_charge):
    """
    Find the CF amplitudes for the events passed in for the
    D -> K+ pi- pi+ pi- (or conjugate, if k_charge is -1)

    NB: arrays should be in MeV; if you have an array in GeV (e.g.
    generated directly by AmpGen), you should scale these yourself
    by multiplying by 1000

    :param k: shape (4, N) array of kaon momenta (px, py, pz, e)
    :param pi1: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi2: shape (4, N) array of pion momenta (px, py, pz, e)
    :param pi3: shape (4, N) array of pion momenta (px, py, pz, e)
    :param k_charge: charge of the kaon, either +1 or -1

    """
    return _amplitudes("cf", k, pi1, pi2, pi3, k_charge)
