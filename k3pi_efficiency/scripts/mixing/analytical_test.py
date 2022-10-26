"""
Plot things that show mixing

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

from lib_efficiency import mixing
import pdg_params


def _plot(params: mixing.MixingParams, path: str, scale: bool = False) -> None:
    """
    Plot the probability of detecting a particle as a D0 or Dbar0 with time

    :param params: mixing parameters
    :path: where to save to
    :scale: whether to scale by dividing probabilities by exponential

    """
    d_lifetime_ps = 0.4103
    times_ps = np.linspace(0, d_lifetime_ps, 500)
    times_inv_mev = times_ps * (10**10) / 6.58

    # Assume no CPV in mixing
    p_q = 1 / np.sqrt(2)

    d0_prob, dbar0_prob = (
        np.abs(amplitude) ** 2
        for amplitude in mixing.mixed_d0_coeffs(
            times=times_inv_mev, p=p_q, q=p_q, params=params
        )
    )

    if scale:
        d0_prob /= np.exp(-times_ps / d_lifetime_ps)
        dbar0_prob /= np.exp(-times_ps / d_lifetime_ps)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_kw = {"linewidth": 1}
    ax.plot(
        times_ps,
        d0_prob,
        label=r"$\frac{p(D^0)}{e^{-t/\tau}}$" if scale else r"$p(D^0)$",
        **plot_kw,
    )
    ax.plot(
        times_ps,
        dbar0_prob,
        label=r"$\frac{p(\overline{D}^0)}{e^{-t/\tau}}$"
        if scale
        else r"$p(\overline{D}^0)$",
        **plot_kw,
    )

    fig.suptitle(f"x={params.mixing_x:.4f}, y={params.mixing_y:.4f}")

    ax.legend()
    ax.set_ylabel("probability")
    ax.set_xlabel("time /ps")

    fig.tight_layout()

    fig.savefig(path)
    plt.clf()


def main():
    """
    Find the D0 and Dbar0 coefficients for particles that begin life as a D

    """
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=pdg_params.mixing_x(),
        mixing_y=pdg_params.mixing_y(),
    )
    _plot(params, "mixing.png", scale=False)
    _plot(params, "scaled_mixing.png", scale=True)

    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=10 * pdg_params.mixing_x(),
        mixing_y=10 * pdg_params.mixing_y(),
    )
    _plot(params, "more_mixing.png", scale=True)


if __name__ == "__main__":
    main()
