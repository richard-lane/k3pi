"""
Find the absolute efficiency for particle gun events

Choose to find these from particle gun, as we have the truth information
readily available

"""
import sys
import pathlib
import argparse
from typing import Tuple
from multiprocessing import Manager, Process
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions
from lib_data import cuts


def _n_reco(data_tree, hlt_tree) -> Tuple[int, int]:
    """
    Number of events reconstructed

    returns number of reco K+ and K-

    """
    # Mask to perform straight cuts
    keep = cuts.sanity_keep(data_tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.hlt_keep_pgun(hlt_tree)

    # Combine with the background category mask
    keep &= cuts.bkgcat(data_tree)

    k_id = data_tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]

    return np.sum(k_id > 0), np.sum(k_id < 0)


def _n_gen(gen_tree) -> Tuple[int, int]:
    """
    Number of events generated

    """
    k_id = gen_tree["D0_P0_ID"].array()

    return np.sum(k_id > 0), np.sum(k_id < 0)


def _efficiency_and_err(n_gen: int, n_reco: int) -> Tuple[float, float]:
    """
    Efficiency and its error (assuming Poisson errors) in %

    """
    efficiency = 100 * n_reco / n_gen

    err = efficiency * np.sqrt((1 / n_reco) + (1 / n_gen))

    return efficiency, err


def _calculate(source_dir, label, out_dict) -> None:
    """
    Calculate efficiency and error for the files in the source dir
    Populate out_dict with label + "_eff" and "label" + "_err"

    """
    # Some folders might break
    broken_folders = []

    n_reco_plus, n_gen_plus = 0, 0
    n_reco_minus, n_gen_minus = 0, 0

    # Iterate over files, incrementing counters
    for i, folder in enumerate(source_dir.glob("*")):
        if not i % 10:
            print(f"{label} {i}")
        try:
            with uproot.open(folder / "pGun_TRACK.root") as data_f, uproot.open(
                folder / "Hlt1TrackMVA.root"
            ) as hlt_f:
                reco_tree = data_f["Dstp2D0pi/DecayTree"]
                gen_tree = data_f["MCDstp2D0pi/MCDecayTree"]
                hlt_tree = hlt_f["DecayTree"]

                # Do cuts on the reco bit, count the entries
                a, b = _n_reco(reco_tree, hlt_tree)
                n_reco_plus += a
                n_reco_minus += a

                # No cuts for the generator level
                a, b = _n_gen(gen_tree)
                n_gen_plus += a
                n_gen_minus += b

        except FileNotFoundError:
            broken_folders.append(str(folder))
            continue

    if broken_folders:
        print(f"{label}: Failed to read from dirs:\n\t{broken_folders}")
        print(
            "This may be expected, e.g. there may be already merged files also in the dir"
        )

    plus_efficiency, plus_err = _efficiency_and_err(n_gen_plus, n_reco_plus)
    minus_efficiency, minus_err = _efficiency_and_err(n_gen_minus, n_reco_minus)

    out_dict[f"plus_{label}_eff"] = plus_efficiency
    out_dict[f"plus_{label}_err"] = plus_err
    out_dict[f"minus_{label}_eff"] = minus_efficiency
    out_dict[f"minus_{label}_err"] = minus_err


def _plot(labels: Tuple, efficiencies: Tuple, errors: Tuple) -> None:
    """
    Plot efficiencies

    TODO labels is unused, so could clean things up a bit by removing it

    """
    indices = list(range(len(labels)))  # wtf
    n = len(indices)

    colours = "r", "g", "b"

    fig, ax = plt.subplots()
    # first half of the list is K+
    for i in indices[: n // 2]:
        ax.errorbar(
            [-2],
            efficiencies[i],
            yerr=errors[i],
            fmt=f"{colours[i]}+",
        )

    # Second half of the list is K-
    for i in indices[n // 2 :]:
        ax.errorbar(
            [2],
            efficiencies[i],
            yerr=errors[i],
            fmt=f"{colours[i - 3]}+",
        )

    patches = [mpatches.Patch(color=c) for c in colours]
    ax.legend(handles=patches, labels=("false", "cf", "dcs"))

    ax.set_xticks([-2, 2])
    ax.set_xlim(-3, 3)
    ax.set_xticklabels([r"$K^+$", r"$K^-$"])
    ax.set_ylabel(r"efficiency /%")

    fig.savefig("efficiencies.png")


def main():
    """
    Open the folder containing the particle gun files that we want
    Open the MC truth and the reconstructed events in each folder
    Apply cuts to the reconstructed events
    Count how many of each type there are
    Report this by just printing it out, for now

    """
    out_dict = Manager().dict()
    labels = ("false", "cf", "dcs")
    procs = [
        Process(target=_calculate, args=(source_dir, label, out_dict))
        for source_dir, label in zip(
            (
                definitions.FALSE_SIGN_SOURCE_DIR,
                definitions.RS_PGUN_SOURCE_DIR,
                definitions.WS_PGUN_SOURCE_DIR,
            ),
            labels,
        )
    ]

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    _plot(
        ["$K^+$" + l for l in labels] + ["$K^-$" + l for l in labels],
        [out_dict[f"plus_{label}_eff"] for label in labels]
        + [out_dict[f"minus_{label}_eff"] for label in labels],
        [out_dict[f"plus_{label}_err"] for label in labels]
        + [out_dict[f"minus_{label}_err"] for label in labels],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the average efficiency for particle gun events"
        "Runs parallel processes for false sign, CF and DCS."
    )
    args = parser.parse_args()

    main()
