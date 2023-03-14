"""
Plot scans of the likelihoods from the charm threshhold experiments

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_time_fit.charm_threshhold import likelihoods


def _ll2chi2(log_likelihoods: np.ndarray) -> np.ndarray:
    """Transform log likelihood to chi2"""
    not_nan = ~np.isnan(log_likelihoods)

    chi2s = np.copy(log_likelihoods)  # transforms to array as well as copying
    chi2s *= -2
    chi2s[not_nan] -= chi2s[not_nan].min()
    chi2s[not_nan] = np.sqrt(chi2s[not_nan])

    return chi2s


def main():
    """
    Choose some values for r_D and y; scan the CLEO likelihood value over the complex plane

    """
    r_d, y, x = 0.0553431, 0.00681448, 0.0036
    bin_number = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    n_re, n_im = 100, 101
    fig, axes = plt.subplot_mosaic(
        "AAABBBCCCD\nAAABBBCCCD\nEEEFFFGGGD\nEEEFFFGGGD\nHHHIIIJJJD\nHHHIIIJJJD\nKKKLLLMMMD\nKKKLLLMMMD",
        figsize=(18, 18),
    )

    contour_kw = {"levels": [0, 1, 2, 3, 4], "cmap": "turbo"}
    bin_axes = [[axes[x] for x in s] for s in ["ABC", "EFG", "HIJ", "KLM"]]

    for bin_number, axes_ in tqdm(zip([0, 1, 2, 3], bin_axes)):
        re_z = np.linspace(-1, 1, n_re)
        im_z = np.linspace(-1, 1, n_im)

        cleo_likelihood_vals = []
        bes_chi2_vals = []
        combined_vals = []
        re_z_vals, im_z_vals = [], []

        cleo_fcn = likelihoods.cleo_fcn()
        bes_fcn = likelihoods.bes_fcn()
        for r in re_z:
            for i in im_z:
                cleo_likelihood_vals.append(
                    likelihoods.cleo(bin_number, r, i, y, r_d, fcn=cleo_fcn)
                )
                bes_chi2_vals.append(
                    likelihoods.bes_chi2(bin_number, r, i, x, y, fcn=bes_fcn)
                )
                combined_vals.append(
                    likelihoods.combined_chi2(
                        bin_number,
                        r,
                        i,
                        x,
                        y,
                        r_d,
                        cleo_fcn=cleo_fcn,
                        bes_fcn=bes_fcn,
                    )
                )
                re_z_vals.append(r)
                im_z_vals.append(i)

        re_z_vals = np.array(re_z_vals)
        im_z_vals = np.array(im_z_vals)

        # Plot CLEO
        axes_[0].contourf(
            re_z_vals.reshape((n_re, n_im)),
            im_z_vals.reshape((n_re, n_im)),
            _ll2chi2(cleo_likelihood_vals).reshape((n_re, n_im)),
            **contour_kw,
        )

        # Plot BES
        bes_chi2_vals = np.array(bes_chi2_vals)
        bes_chi2_vals -= bes_chi2_vals.min()
        axes_[1].contourf(
            re_z_vals.reshape((n_re, n_im)),
            im_z_vals.reshape((n_re, n_im)),
            bes_chi2_vals.reshape((n_re, n_im)),
            **contour_kw,
        )

        # Plot combined
        combined_vals = np.array(combined_vals)
        not_nan = ~np.isnan(combined_vals)
        combined_vals[not_nan] -= combined_vals[not_nan].min()
        contours = axes_[2].contourf(
            re_z_vals.reshape((n_re, n_im)),
            im_z_vals.reshape((n_re, n_im)),
            combined_vals.reshape((n_re, n_im)),
            **contour_kw,
        )

        for axis in axes_:
            axis.add_patch(Circle((0, 0), radius=1, facecolor="none", edgecolor="k"))
            axis.set_aspect("equal")

    axes["A"].set_ylabel("bin 1")
    axes["E"].set_ylabel("bin 2")
    axes["H"].set_ylabel("bin 3")
    axes["K"].set_ylabel("bin 4")

    axes["A"].set_title("CLEO")
    axes["B"].set_title("BES-III")
    axes["C"].set_title("CLEO + BES-III")

    fig.colorbar(contours, cax=axes["D"])
    fig.savefig("charm_threshhold_chi2s.png")


if __name__ == "__main__":
    main()
