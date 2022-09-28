"""
Example script perhaps

"""
import os
import sys
import glob
import tqdm
import pickle
import uproot
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from hep_ml.uboost import uBoostBDT
from iminuit.cost import ExtendedUnbinnedNLL

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from libFit import util
from libFit import pdfs
from libFit import fit


def _data_branches():
    return (
        "Dst_ReFit_chi2",  # done
        "D0_ENDVERTEX_CHI2",  # done
        "D0_ORIVX_CHI2",  # done
        "Dst_pi_ProbNNpi",  # done
        "D0_PT",  # should be ReFit?
        "Dst_ReFit_piplus_PT",  # Need to calculate
        "Dst_pi_IPCHI2_OWNPV",  # done
        "pion_PT_max",  # Need to calculate
        "pion_PT_min",  # Need to calculate
        "pion_PT_sum",  # Need to calculate
        "PT_max",  # Need to calculate
        "PT_min",  # Need to calculate
        "PT_sum",  # Need to calculate - might be D0_Loki_SUMPT
    )


def _norm(*arrays) -> np.array:
    """
    Euclidean norm

    """
    rv = np.zeros_like(arrays[0], dtype=np.float64)

    for a in arrays:
        tmp = a.astype(np.float64)
        rv = rv + tmp**2

    return np.sqrt(rv)


def _read_data(sign: str) -> pd.DataFrame:
    """
    Read a dataframe containing our training features from real data

    """
    data_dir = "/home/mh19137/Documents/HEP/data/data/tmp/*.root"
    dfs = []
    tree_name = (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )
    for path in tqdm.tqdm(glob.glob(data_dir)):
        tmp = pd.DataFrame()
        with uproot.open(path) as f:
            tree = f[tree_name]

            tmp["D0_M_ReFit"] = tree["Dst_ReFit_D0_M"].array()[:, 0]
            tmp["Delta_M"] = tree["Dst_ReFit_M"].array()[:, 0] - tmp["D0_M_ReFit"]

            for i, (data_branch, training_branch) in enumerate(
                zip(_data_branches(), util.train_features())
            ):
                if i == 0:
                    # Jagged array
                    tmp[training_branch] = tree[data_branch].array()[:, 0]
                    continue

                # First 5 branches are literally in the tree
                # TODO maybe we need to refit the D0_PT, in which case only the first 4 are
                if i < 5 or i == 6:
                    tmp[training_branch] = tree[data_branch].array()
                    continue

                # Refit piplus pT
                if i == 5:
                    piplus_px = tree["Dst_ReFit_piplus_PX"].array()[:, 0]
                    piplus_py = tree["Dst_ReFit_piplus_PY"].array()[:, 0]
                    tmp[training_branch] = _norm(piplus_px, piplus_py)
                    continue

                pion_px = tuple(
                    tree[b].array()[:, 0]
                    for b in (
                        "Dst_ReFit_D0_piplus_0_PX",
                        "Dst_ReFit_D0_piplus_1_PX",
                        "Dst_ReFit_D0_piplus_PX",
                    )
                )
                pion_py = tuple(
                    tree[b].array()[:, 0]
                    for b in (
                        "Dst_ReFit_D0_piplus_0_PY",
                        "Dst_ReFit_D0_piplus_1_PY",
                        "Dst_ReFit_D0_piplus_PY",
                    )
                )

                pion_pt = tuple(_norm(*x) for x in zip(pion_px, pion_py))

                if i == 7:
                    tmp[training_branch] = np.amax(pion_pt, axis=0)
                    continue

                if i == 8:
                    tmp[training_branch] = np.amin(pion_pt, axis=0)
                    continue

                if i == 9:
                    tmp[training_branch] = np.sum(pion_pt, axis=0)
                    continue

                px = (*pion_px, tree["Dst_ReFit_D0_Kplus_PX"].array()[:, 0])
                py = (*pion_py, tree["Dst_ReFit_D0_Kplus_PY"].array()[:, 0])
                pt = tuple(_norm(*x) for x in zip(px, py))

                if i == 10:
                    tmp[training_branch] = np.amax(pt, axis=0)
                    continue

                if i == 11:
                    tmp[training_branch] = np.amin(pt, axis=0)
                    continue

                if i == 12:
                    tmp[training_branch] = np.sum(pt, axis=0)
                    continue

        tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp.dropna(inplace=True)
        dfs.append(tmp)

    return pd.concat(dfs)


def _newest_classifier() -> uBoostBDT:
    """
    Get the newest classifier

    TODO check type

    """
    classifiers = list(glob.glob("*.pkl"))
    classifiers.sort(key=lambda x: os.path.getmtime(x))
    with open(classifiers[0], "rb") as f:
        clf = pickle.load(f)
        assert isinstance(clf, uBoostBDT)
        return clf


def _plot(data, mc, path):
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    kw = {"histtype": "step", "density": True}
    for a, label in zip(ax.ravel(), data.keys()):
        lo, hi = np.quantile(mc[label], [0.05, 0.95])
        bins = np.linspace(0.9 * lo, 1.1 * hi, 100)
        a.hist(mc[label], label="MC", bins=bins, **kw)
        a.hist(data[label], label="data", bins=bins, **kw)
        a.set_title(label)
        a.legend()

    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main():
    # Read MC and data into pandas dataframes
    mc = util.read_dataframe()
    sign = "WS"
    data = _read_data(sign)

    # Plot all the variables in both
    _plot(data, mc, "mc_data_all.png")

    # Plot only the events the classifier things are signal
    clf = _newest_classifier()
    data_signal = clf.predict(data) == 1
    mc_signal = clf.predict(mc) == 1

    _plot(data[data_signal], mc[mc_signal], "mc_data_sig.png")

    # Do a mass fit to the thing we just removed background from with the BDT
    fitter = fit.fit(data["Delta_M"].to_numpy(), sign, 5, 0.8)


if __name__ == "__main__":
    main()
