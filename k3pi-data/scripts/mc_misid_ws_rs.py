"""
Find out how often we misreconstruct a RS event as WS
by swapping a kaon and pion

"""
import sys
import pathlib
import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions
from lib_data import cuts


def _keep_no_pid(tree):
    """
    D0, delta M, ipchi2 and trigger cuts

    """
    return np.logical_and.reduce(
        [
            fcn(tree)
            for fcn in (
                cuts._d0_mass_keep,
                cuts._delta_m_keep,
                cuts._ipchi2_keep,
                cuts._trigger_keep,
            )
        ]
    )


def _num_swapped(k_id: np.ndarray, pi1_id: np.ndarray, pi3_id: np.ndarray) -> int:
    """
    Number of evts we've swapped the K->pi and at least one of the pis -> K

    """
    k, pi = 321, 211
    return np.sum(((k_id == pi) & (pi1_id == k)) | ((k_id == pi) & (pi3_id == k)))


def main():
    """
    Read the WS MC, look at true PID for daughters
    If we have misid'd a pion as the kaon, and the kaon as any of the pions
    then we're in trouble

    See the effect of the pid cut on this

    """
    year, magnetisation = "2018", "magdown"

    # Interested in RS being misIDd as WS
    sign = "cf"

    n_before, n_cut, n_pid = 0, 0, 0
    tot_before, tot_cut, tot_pid = 0, 0, 0

    pi_probs = [[] for _ in range(4)]
    k_probs = [[] for _ in range(4)]

    cut_keep = []
    pid_keep = []

    for path in tqdm(definitions.mc_files(year, magnetisation, sign)):
        with uproot.open(path) as mc_f:
            tree = mc_f[definitions.data_tree(sign)]

            # Cuts
            cut_keep.append(_keep_no_pid(tree))
            pid_keep.append(cut_keep[-1] & cuts._pid_keep(tree))

            k_true_id = np.abs(tree["D0_P0_TRUEID"].array(library="np"))
            pi1_true_id = np.abs(tree["D0_P1_TRUEID"].array(library="np"))
            pi3_true_id = np.abs(tree["D0_P3_TRUEID"].array(library="np"))

            n_before += _num_swapped(k_true_id, pi1_true_id, pi3_true_id)
            n_cut += _num_swapped(
                k_true_id[cut_keep[-1]],
                pi1_true_id[cut_keep[-1]],
                pi3_true_id[cut_keep[-1]],
            )
            n_pid += _num_swapped(
                k_true_id[pid_keep[-1]],
                pi1_true_id[pid_keep[-1]],
                pi3_true_id[pid_keep[-1]],
            )

            for i in range(4):
                pi_probs[i].append(tree[f"D0_P{i}_ProbNNpi"].array(library="np"))
                k_probs[i].append(tree[f"D0_P{i}_ProbNNk"].array(library="np"))

            tot_before += len(cut_keep[-1])
            tot_cut += np.sum(cut_keep[-1])
            tot_pid += np.sum(pid_keep[-1])

    print(n_before, n_cut, n_pid)
    print(100 * n_before / tot_before, 100 * n_cut / tot_cut, 100 * n_pid / tot_pid)

    pi_probs = [np.concatenate(prob) for prob in pi_probs]
    k_probs = [np.concatenate(prob) for prob in k_probs]
    cut_keep = np.concatenate(cut_keep)
    pid_keep = np.concatenate(pid_keep)

    fig, ax = plt.subplots(4, 3, figsize=(9, 12))
    hist_kw = {"bins": np.linspace(0, 1, 100), "histtype": "step", "density": False}
    for axis, pi_prob, k_prob in zip(ax, pi_probs, k_probs):
        probs = (pi_prob, k_prob, pi_prob * (1 - k_prob))
        for ax_, prob in zip(axis, probs):
            ax_.hist(prob, **hist_kw, label="All")
            ax_.hist(prob[cut_keep], **hist_kw, label="Cuts without PID")
            ax_.hist(prob[pid_keep], **hist_kw, label="Cuts including PID")
            ax_.set_yticks([])
            ax_.set_yscale("log")

    for axis, label in zip(
        ax[0], (r"ProbNN$\pi$", r"ProbNN$K$", r"ProbNN$\pi$ $\times$ ProbNN$K$")
    ):
        axis.set_title(label)

    for axis, label in zip(
        ax[:, 0], (r"K$^\pm$", r"$\pi^\mp$", r"$\pi^\mp$", r"$\pi^\pm$")
    ):
        axis.set_ylabel(label)

    ax[0, -1].legend()
    fig.tight_layout()
    fig.savefig("mc_pid.png")
    plt.show()


if __name__ == "__main__":
    main()
