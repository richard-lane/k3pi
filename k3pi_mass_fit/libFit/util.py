"""
Utilitility fractions

"""
import os
import sys
import pathlib
from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd
from iminuit.util import ValueView

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi_efficiency"))

from lib_data import stats, get
from lib_data.util import k_3pi
from lib_cuts.get import cut_dfs, classifier as get_clf
from lib_efficiency.get import reweighter_dump as get_reweighter
from lib_efficiency.efficiency_util import points


def delta_m(dataframe: pd.DataFrame) -> pd.Series:
    """
    Get mass difference from a dataframe

    """
    return dataframe["D* mass"] - dataframe["D0 mass"]


def delta_m_generator(dataframes: Iterable[pd.DataFrame]) -> Iterable[pd.Series]:
    """
    Get generator of mass differences from an iterable of dataframes

    """
    for dataframe in dataframes:
        yield delta_m(dataframe)


def delta_m_counts(
    dataframes: Iterable[pd.DataFrame],
    bins: np.ndarray,
    weights: Iterable[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return binned delta M from an iterable of dataframes

    """
    return stats.counts_generator(delta_m_generator(dataframes), bins, weights)


def binned_delta_m_counts(
    dataframes: Iterable[pd.DataFrame],
    mass_bins: np.ndarray,
    n_time_bins: int,
    time_indices: Iterable[np.ndarray],
    weights: Iterable[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return binned delta M from an iterable of dataframes

    """
    return stats.time_binned_counts(
        delta_m_generator(dataframes), mass_bins, n_time_bins, time_indices, weights
    )


def rs_ws_params(params: ValueView) -> Tuple[Tuple, Tuple]:
    """
    Find RS and WS params from binned simultaneous fitter params

    """
    rs_params = (*params[:2], *params[4:-2])
    ws_params = (*params[2:-4], *params[-2:])

    return rs_params, ws_params


def fit_params(rs_params: ValueView, ws_params: ValueView) -> Tuple:
    """
    Find fitter parameters from RS and WS params

    i.e. the inverse of the above

    """
    # Check the RS and WS signal params are the same
    for i in range(2, 7):
        assert rs_params[i] == ws_params[i]

    return (*rs_params[:2], *ws_params[:2], *rs_params[2:], *ws_params[-2:])


def alt_rs_ws_params(params: ValueView) -> Tuple[Tuple, Tuple]:
    """
    Find RS and WS params from binned simultaneous fitter params
    using the alternate bkg model

    """
    rs_params = (*params[:2], *params[4:13])
    ws_params = (*params[2:10], *params[13:])

    return rs_params, ws_params


def _generators(
    year: str, magnetisation: str, *, bdt_cut: bool, phsp_bin: int
) -> Tuple[Iterable[pd.DataFrame], Iterable[pd.DataFrame]]:
    """
    Generator of rs/ws dataframes, with/without BDT cut

    """
    generators = [
        get.binned_generator(get.data(year, sign, magnetisation), phsp_bin)
        for sign in ("cf", "dcs")
    ]

    if bdt_cut:
        # Get the classifier
        # always use DCS BDT for cut
        clf = get_clf(year, "dcs", magnetisation)
        return [cut_dfs(gen, clf) for gen in generators]

    return generators


def _time_indices(
    dataframes: Iterable[pd.DataFrame], time_bins: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Generator of time bin indices from a generator of dataframes

    """
    return stats.bin_indices((dataframe["time"] for dataframe in dataframes), time_bins)


def _efficiency_generators(
    year: str,
    magnetisation: str,
    *,
    phsp_bin: int,
) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
    """
    Generator of efficiency weights

    """
    # Do BDT cut
    cf_gen, dcs_gen = _generators(year, magnetisation, bdt_cut=True, phsp_bin=phsp_bin)

    # Open efficiency weighters
    dcs_weighter = get_reweighter(
        year, "dcs", magnetisation, "both", fit=False, cut=True, verbose=True
    )
    cf_weighter = get_reweighter(
        year, "cf", magnetisation, "both", fit=False, cut=True, verbose=True
    )

    # Generators to get weights
    return (
        (
            cf_weighter.weights(points(*k_3pi(dataframe), dataframe["time"]))
            for dataframe in cf_gen
        ),
        (
            dcs_weighter.weights(points(*k_3pi(dataframe), dataframe["time"]))
            for dataframe in dcs_gen
        ),
    )


def mass_counts(
    year: str,
    magnetisation: str,
    mass_bins: np.ndarray,
    time_bins: np.ndarray,
    *,
    bdt_cut: bool,
    correct_efficiency: bool,
    phsp_bin: int,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Find lists of arrays of counts and errors in each time bin

    :param year: data taking year
    :param magnetisation: magnetisation direction
    :param mass_bins: mass fit bins
    :param time_bins: time indexing bins
    :param bdt_cut: whether to perform the BDT cut too
    :param correct_efficiency: whether to apply efficiency correction weights
    :param phsp_bin: which phsp bin number to use
    :param verbose: whether to print stuff

    :returns: list of arrays of WS counts in each time bin;
              one list for each time bin
    :returns: list of arrays of RS counts in each time bin;
              one list for each time bin
    :returns: list of arrays of WS errors in each time bin;
              one list for each time bin
    :returns: list of arrays of RS errors in each time bin;
              one list for each time bin

    """
    # Get generators of time indices
    if verbose:
        print("getting generators of time bin indices")

    cf_gen, dcs_gen = _generators(
        year, magnetisation, bdt_cut=bdt_cut, phsp_bin=phsp_bin
    )
    dcs_indices = _time_indices(dcs_gen, time_bins)
    cf_indices = _time_indices(cf_gen, time_bins)

    # Need to get new generators now that we've used them up
    cf_gen, dcs_gen = _generators(
        year, magnetisation, bdt_cut=bdt_cut, phsp_bin=phsp_bin
    )

    # May need to get generators of efficiency weights too
    if correct_efficiency:
        if verbose:
            print("getting generators of efficiency weights")
        cf_wts, dcs_wts = _efficiency_generators(year, magnetisation, phsp_bin=phsp_bin)
    else:
        cf_wts, dcs_wts = None, None

    n_time_bins = len(time_bins) - 1

    if verbose:
        print("getting DCS counts")
    dcs_counts, dcs_errs = binned_delta_m_counts(
        dcs_gen, mass_bins, n_time_bins, dcs_indices, dcs_wts
    )

    if verbose:
        print("getting CFcounts")
    cf_counts, cf_errs = binned_delta_m_counts(
        cf_gen, mass_bins, n_time_bins, cf_indices, cf_wts
    )

    return dcs_counts, cf_counts, dcs_errs, cf_errs


def plot_dir(bdt_cut: bool, correct_efficiency: bool, phsp_bin: int) -> str:
    """
    Directory for storing plots

    / terminated

    Creates it if it doesnt exist

    Also creates subdirs /ws/ and /rs/
    and some alt bkg subdirs

    """
    if correct_efficiency:
        assert bdt_cut

    if bdt_cut and correct_efficiency:
        retval = "eff_fits/"
    elif bdt_cut:
        retval = "bdt_fits/"
    else:
        retval = "raw_fits/"

    retval = os.path.join(retval, f"bin_{phsp_bin}/")

    for dir_ in (
        retval,
        f"{retval}ws/",
        f"{retval}rs/",
        f"{retval}alt_bkg_rs",
        f"{retval}alt_bkg_ws",
    ):
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

    return retval


def sqrt_bkg_param_guess(sign: str = "dcs") -> Tuple:
    """
    Some parameters for the sqrt background

    """
    assert sign in {"dcs", "cf"}
    if sign == "dcs":
        return -0.02, 0.0

    return -0.016, -0.0002


def signal_param_guess(time_bin: int = 5) -> Tuple:
    """
    Some parameters for the signal

    """
    return 146.0, 0.2, 0.2, 0.18, 0.18, 0.0019 * time_bin + 0.0198


def yield_file(
    year: str, magnetisation: str, phsp_bin: int, bdt_cut: bool, efficiency: bool
) -> pathlib.Path:
    """
    For writing the yields to

    """
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / f"yields_{year}_{magnetisation}_{phsp_bin}_{bdt_cut=}_{efficiency=}.txt"
    )


def write_yield(
    times: Tuple,
    yields: Tuple,
    errs: Tuple,
    path: pathlib.Path,
    *,
    print_str: bool = False,
) -> None:
    """
    Append yields to a file

    :param times: tuple (low time, high time) used for the fit
    :param yields: tuple (RS yield, WS yield)
    :param errs: tuple (RS err, WS err)
    :param path: where the file lives. it should already exist
    :param print_str: print the string that will be written

    """
    assert path.exists()

    out_str = "\t".join((str(num) for num in (*times, *yields, *errs)))
    out_str += "\n"
    if print_str:
        print(out_str)

    with open(str(path), "a", encoding="utf8") as yield_f:
        yield_f.write(out_str)


def read_yield(path: pathlib.Path) -> Tuple:
    """
    Read time bins, RS yields, RS errs, WS yields, WS errs
    from a file

    Returns arrays

    """
    time_bins = []
    rs_yields = []
    rs_errs = []
    ws_yields = []
    ws_errs = []

    with open(str(path), "r", encoding="utf8") as yield_f:
        for line in yield_f.readlines():
            low_t, high_t, rs_yield, rs_err, ws_yield, ws_err = (
                float(val) for val in line.strip().split("\t")
            )

            # Add the high time bin from each line, except the first
            # line in which case we add both
            if not time_bins:
                time_bins.append(low_t)
            time_bins.append(high_t)

            rs_yields.append(rs_yield)
            rs_errs.append(rs_err)

            ws_yields.append(ws_yield)
            ws_errs.append(ws_err)

    return tuple(
        np.array(arr) for arr in (time_bins, rs_yields, rs_errs, ws_yields, ws_errs)
    )
