"""
Manual test for the amplitude models

Check that if we scale the RS and WS ampgen models by the amplitudes that they look the same
(and they look like phase space, but I don't directly check this).

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import get, util
from lib_efficiency.amplitude_models import amplitudes
from lib_efficiency import efficiency_util


def main():
    """
    Read the ampgen files, find the amplitudes for each event, weight each event by 1/amplitude^2
    and plot

    """
    n_evts = 2000000  # Only use the first n evts otherwise my computer crashes
    ws_df = get.ampgen("dcs")[:n_evts]
    rs_df = get.ampgen("cf")[:n_evts]

    # tuple of (k, pi1, pi2, pi3) for each
    ws_k, ws_pi1, ws_pi2, ws_pi3 = efficiency_util.k_3pi(ws_df)
    rs_k, rs_pi1, rs_pi2, rs_pi3 = efficiency_util.k_3pi(rs_df)

    # Kaon charge is +1
    ws_amp = amplitudes.dcs_amplitudes(ws_k, ws_pi1, ws_pi2, ws_pi3, +1)
    rs_amp = amplitudes.cf_amplitudes(rs_k, rs_pi1, rs_pi2, rs_pi3, +1)

    ws_weights = 1 / (np.abs(ws_amp) ** 2)
    rs_weights = 1 / (np.abs(rs_amp) ** 2)

    ws_weights /= np.mean(ws_weights)
    rs_weights /= np.mean(rs_weights)

    ws_pi1, ws_pi2 = util.momentum_order(ws_k, ws_pi1, ws_pi2)
    rs_pi1, rs_pi2 = util.momentum_order(rs_k, rs_pi1, rs_pi2)

    ws = np.column_stack((helicity_param(ws_k, ws_pi1, ws_pi2, ws_pi3), ws_df["time"]))
    rs = np.column_stack((helicity_param(rs_k, rs_pi1, rs_pi2, rs_pi3), rs_df["time"]))

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    for axis, rs_x, ws_x in zip(ax.ravel(), rs.T, ws.T):
        _, bins, _ = axis.hist(rs_x, bins=100, label="RS", color="b", histtype="step")
        axis.hist(ws_x, bins=bins, label="WS", color="r", histtype="step")

        axis.hist(
            rs_x,
            bins=bins,
            label="RS scaled",
            color="b",
            alpha=0.4,
            weights=rs_weights,
        )
        axis.hist(
            ws_x,
            bins=bins,
            label="WS scaled",
            color="r",
            alpha=0.4,
            weights=ws_weights,
        )

    ax[0, 0].legend()

    fig.savefig("tmp.png")

    plt.show()


if __name__ == "__main__":
    main()
