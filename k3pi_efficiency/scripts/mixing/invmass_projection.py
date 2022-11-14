"""
Based on a demonstration that Tim showed,
plot a projection of K+pi- mass
as time goes on and Ds mix

bad script full of magic numbers

"""
import sys
import pathlib
from typing import Tuple
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))
import pdg_params
from lib_efficiency import efficiency_util, mixing
from lib_data.definitions import MOMENTUM_SUFFICES

# I don't like it but apparently the matplotlib
# animation should be a global variable
ANI = None


def _params():
    """mixing parameters"""
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=1000 * pdg_params.mixing_x(),
        mixing_y=10 * pdg_params.mixing_y(),
    )
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    times = np.linspace(0, 5, 64)

    return params, q_p, times


def _weights(k3pi: Tuple, times: np.ndarray) -> np.ndarray:
    """
    Add mixing to the dataframe at the provided time

    """
    params, q_p, _ = _params()
    print(".", end="", flush=True)

    retval = mixing.ws_mixing_weights(
        k3pi,
        times,
        params,
        +1,
        q_p,
    )
    return retval


def _counts(masses, weights=None):
    bins = np.linspace(600, 1600, 100)
    centres = (bins[1:] + bins[:-1]) / 2
    counts, _ = np.histogram(masses, bins=bins, density=True, weights=weights)

    return centres, counts


def _plot(
    axis: plt.Axes, masses: np.ndarray, label: str, weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns counts

    """
    axis.plot(*_counts(masses, weights), label=label)


def _invmass(dataframe: pd.DataFrame) -> np.ndarray:
    """
    K+pi- invariant masses from dataframe

    """
    momenta = np.array(
        tuple(
            dataframe[f"Kplus_{s}"] + dataframe[f"pi3plus_{s}"]
            for s in MOMENTUM_SUFFICES
        )
    )

    return np.sqrt(momenta[-1] ** 2 - np.sum(momenta[:-1], axis=0))


def main():
    """
    Read AmpGen dataframes,
    add mixing via the models
    plot projections as a function of time,
    save as a gif

    """
    # Read AmpGen dataframes
    num = None
    cf_df = efficiency_util.ampgen_df("cf", "k_plus", train=None)[:num]
    dcs_df = efficiency_util.ampgen_df("dcs", "k_plus", train=None)[:num]

    # Plot projection of inv mass at each time
    cf_mass = _invmass(cf_df)
    dcs_mass = _invmass(dcs_df)
    fig, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB")

    _plot(axes["A"], cf_mass, "CF", None)
    _plot(axes["A"], dcs_mass, "DCS", None)
    axes["A"].set_xlabel(r"M(K+$\pi+$)")
    axes["B"].set_xlabel(r"t/ps")

    params, q_p, times = _params()
    fig.suptitle(f"{params.mixing_x=:.2f} {params.mixing_y=:.2f}")

    # Plot D probabilities on an axis
    times_inv_mev = times * (10**10) / 6.58
    d0_prob, dbar0_prob = (
        np.abs(amplitude) ** 2
        for amplitude in mixing.mixed_d0_coeffs(
            times=times_inv_mev, p=q_p[0], q=q_p[1], params=params
        )
    )
    axes["B"].plot(times, d0_prob)
    axes["B"].plot(times, dbar0_prob)
    # axes["B"].set_yscale("log")

    # Add mixing at various times
    # We don't really care about the overall scale, just how the
    # shape changes
    # So don't attempt to scale the amplitudes by the right relative amounts or anything
    # This would be required if we wanted to make the mixing realistic
    k3pi = efficiency_util.k_3pi(dcs_df)
    print("." * len(times))
    with Pool(processes=8) as pool:
        weights = pool.starmap(
            _weights,
            zip((k3pi for _ in times), (time * np.ones(len(dcs_df)) for time in times)),
        )
        print()

    (line,) = axes["A"].plot([], [], "k-", lw=2)
    (dcs_point,) = axes["B"].plot([], [], "ko", markersize=2, label=r"p($D^0$)")
    (cf_point,) = axes["B"].plot(
        [], [], "ko", markersize=2, label=r"p($\overline{D}^0$)"
    )
    axes["A"].legend()

    def init():
        line.set_data(*_counts(dcs_mass, weights[0]))
        line.set_label("mixing")
        dcs_point.set_data([0, 1])
        cf_point.set_data([0, 0])
        return (line, dcs_point, cf_point)

    def animate(i):
        line.set_data(*_counts(dcs_mass, weights[i]))

        dcs_point.set_data(times[i], d0_prob[i])
        cf_point.set_data(times[i], dbar0_prob[i])
        return (line, dcs_point, cf_point)

    # No idea why this has to be global
    global ANI
    ANI = animation.FuncAnimation(
        fig,
        animate,
        np.arange(0, len(weights)),
        interval=100,
        blit=True,
        init_func=init,
    )
    axes["A"].set_ylim(0, 0.005)
    ANI.save("invmass_mixing.gif", writer="imagemagick", fps=12)

    plt.show()


if __name__ == "__main__":
    main()
