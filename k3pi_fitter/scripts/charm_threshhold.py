"""
Plot scans of the likelihoods from the charm threshhold experiments

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_time_fit.charm_threshhold import likelihoods


def main():
    """
    Choose some values for r_D and y; scan the CLEO likelihood value over the complex plane

    """
    r_d, y = 0.0553431, 0.00681448
    bin_number = 3

    n_re, n_im = 200, 201
    re_z = np.linspace(-1, 1, n_re)
    im_z = np.linspace(-1, 1, n_im)

    likelihood_vals = []
    re_z_vals, im_z_vals = [], []

    cleo_fcn = likelihoods.cleo_fcn()
    for r in re_z:
        for i in im_z:
            likelihood_vals.append(
                likelihoods.cleo(bin_number, r, i, y, r_d, fcn=cleo_fcn)
            )
            re_z_vals.append(r)
            im_z_vals.append(i)

    re_z_vals = np.array(re_z_vals)
    im_z_vals = np.array(im_z_vals)
    # Transform log likelihood to chi2
    likelihood_vals = np.array(likelihood_vals)
    not_nan = ~np.isnan(likelihood_vals)
    likelihood_vals *= -2
    likelihood_vals[not_nan] -= likelihood_vals[not_nan].min()
    likelihood_vals[not_nan] = np.sqrt(likelihood_vals[not_nan])

    fig, ax = plt.subplots()
    contours = ax.contourf(
        re_z_vals.reshape((n_re, n_im)),
        im_z_vals.reshape((n_re, n_im)),
        likelihood_vals.reshape((n_re, n_im)),
        levels=[0, 1, 2, 3, 4],
        cmap="turbo",
    )
    fig.colorbar(contours, ax=ax)
    fig.savefig(f"cleo_bin_{bin_number}.png")


if __name__ == "__main__":
    main()
