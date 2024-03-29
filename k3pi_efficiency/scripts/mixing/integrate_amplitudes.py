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


def _amplitudes(gen, n_gen: int, k_charge: int):
    """
    Generate n_gen points, keep some of them, find average amplitude model

    Returns num

    """
    # Generate some points
    k, pi1, pi2, pi3 = _gen(gen, n_gen)

    # Evaluate amplitude at each
    cf = amplitudes.cf_amplitudes(k, pi1, pi2, pi3, k_charge)
    dcs = amplitudes.dcs_amplitudes(k, pi1, pi2, pi3, k_charge)

    # Find cross term
    cross = cf * dcs.conjugate()

    # Find sum of squares
    cf = np.abs(cf) ** 2
    dcs = np.abs(dcs) ** 2

    return len(cf), np.sum(cf), np.sum(dcs), np.sum(cross)


def main():
    """
    Generate phsp evts, evaluate |amplitudes|^2

    """
    k_charge = -1

    # Generate lots of phase space events
    n_gen = 500000

    n_tot = 0
    cf_sum = 0
    dcs_sum = 0
    cross = 0.0 + 0.0j

    # We only want to initialise our generator once
    k_mass = 493.677
    pi_mass = 139.570
    d_mass = 1864.84
    gen = ps.nbody_decay(
        d_mass, (k_mass, pi_mass, pi_mass, pi_mass), names=["k", "pi1", "pi2", "pi3"]
    )

    try:
        for _ in range(1000):
            retval = _amplitudes(gen, n_gen, k_charge)
            n_tot += retval[0]
            cf_sum += retval[1]
            dcs_sum += retval[2]
            cross += retval[3]

            cf_avg = cf_sum / n_tot
            dcs_avg = dcs_sum / n_tot
            z = cross / np.sqrt(cf_sum * dcs_sum)

            print(f"{n_tot: <10,}\t{cf_avg:.10f}\t{dcs_avg:.10f}\t{z:.10f}")
    except KeyboardInterrupt:
        ...


if __name__ == "__main__":
    main()
