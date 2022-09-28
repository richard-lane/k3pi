"""
Generate lots of points uniformly in phase space

Evaluate the amplitude at each to find the integral of the amplitudes

"""
import sys
import pathlib
import numpy as np
import phasespace as ps

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from lib_efficiency.amplitude_models import amplitudes


def _gen(gen, n_gen: int):
    """
    Generate k3pi

    """
    weights, particles = gen.generate(n_gen, normalize_weights=True)

    keep = (np.max(weights) * np.random.random(n_gen)) < weights

    return (particles[name].numpy()[keep].T for name in ["k", "pi1", "pi2", "pi3"])


def _amplitudes(gen, n_gen: int):
    """
    Generate n_gen points, keep some of them, find average amplitude model

    Returns num

    """
    # Generate some points
    k, pi1, pi2, pi3 = _gen(gen, n_gen)

    # Evaluate amplitude at each
    cf = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, +1)
    dcs = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, +1)

    # Find sum of squares
    cf = np.abs(cf) ** 2
    dcs = np.abs(dcs) ** 2

    return len(cf), np.sum(cf), np.sum(dcs)


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

    """
    # Generate lots of phase space events
    n_gen = 10000000

    n_tot = 0
    cf_sum = 0
    dcs_sum = 0

    # We only want to initialise our generator once
    k_mass = 493.677
    pi_mass = 139.570
    d_mass = 1864.84
    gen = ps.nbody_decay(
        d_mass, (k_mass, pi_mass, pi_mass, pi_mass), names=["k", "pi1", "pi2", "pi3"]
    )

    for _ in range(10):
        retval = _amplitudes(gen, n_gen)
        n_tot += retval[0]
        cf_sum += retval[1]
        dcs_sum += retval[2]

        print(f"{n_tot: <10,}\t{cf_sum / n_tot:.6f}\t{dcs_sum / n_tot:.6f}")


if __name__ == "__main__":
    main()
