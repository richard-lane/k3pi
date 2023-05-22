"""
Plot scans of the likelihoods from the charm threshhold experiments

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_time_fit import parabola, plotting, definitions
from lib_time_fit.charm_threshhold import likelihoods


def _ll2chi2(log_likelihoods: np.ndarray) -> np.ndarray:
    """Transform log likelihood to chi2"""
    chi2s = log_likelihoods * -2
    chi2s -= np.nanmin(chi2s)
    return chi2s


def main():
    """
    Choose some values for r_D and y; scan the CLEO likelihood value over the complex plane

    """
    # r_d, y, x = 0.0553431, 0.00681448, 0.0036
    r_d, y, x = 0.0553431, definitions.CHARM_Y, definitions.CHARM_X
    bin_number = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    n_re, n_im = 100, 101
    re_z = np.linspace(-1, 1, n_re)
    im_z = np.linspace(-1, 1, n_im)

    cleo_likelihood_vals = np.ones((n_im, n_re))
    bes_chi2_vals = np.ones((n_im, n_re))
    combined_vals = np.ones((n_im, n_re))

    cleo_fcn = likelihoods.cleo_fcn()
    bes_fcn = likelihoods.bes_fcn()
    for i, re in enumerate(re_z):
        for j, im in enumerate(im_z):
            cleo_likelihood_vals[j, i] = likelihoods.cleo(
                bin_number, re, im, y, r_d, fcn=cleo_fcn
            )
            bes_chi2_vals[j, i] = likelihoods.bes_chi2(
                bin_number, re, im, x, y, fcn=bes_fcn
            )
            combined_vals[j, i] = likelihoods.combined_chi2(
                bin_number,
                re,
                im,
                x,
                y,
                r_d,
                cleo_fcn=cleo_fcn,
                bes_fcn=bes_fcn,
            )

    # Transform CLEO LL to chi2
    cleo_chi2_vals = _ll2chi2(cleo_likelihood_vals)

    # Subtract the minimum off the others
    bes_chi2_vals -= np.nanmin(bes_chi2_vals)
    combined_vals -= np.nanmin(combined_vals)

    # Find the best fit value of Z
    min_im, min_re = np.unravel_index(np.nanargmin(combined_vals), combined_vals.shape)
    best_z = re_z[min_re], im_z[min_im]
    print(f"{best_z=}")

    # Fit a parabola to the combined chi2 and report the parameters
    max_chi2 = 9
    params, errs = parabola.fit(combined_vals, re_z, im_z, best_z, max_chi2)
    for param, err, label in zip(
        params,
        errs,
        ["ReZ", "ImZ", "ReZ widthL", "ReZ width R", "imz w L", "imz w R", "corr"],
    ):
        print(f"{label}\t= {param:.3f} +- {err:.3f}")

    fig, axes = plotting.projections((re_z, im_z), combined_vals)
    parabola.plot_projection(axes, params, max_chi2)
    axes[0].set_ylim(0, 16)
    axes[0].legend()
    fig.savefig(f"charm_combination_parabola_{bin_number}.png")
    plt.close(fig)

    # Transform from chi2s to sigma
    cleo_chi2_vals = np.sqrt(cleo_chi2_vals)
    bes_chi2_vals = np.sqrt(bes_chi2_vals)
    combined_vals = np.sqrt(combined_vals)

    # Plot sigma bands
    fig, axes = plt.subplot_mosaic(
        "AAABBBCCCD\nAAABBBCCCD\nAAABBBCCCD", figsize=(19, 6)
    )
    levels = [0, 1, 2, 3]
    contour_kw = {
        "colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(levels)]
    }
    plotting.scan(axes["A"], re_z, im_z, cleo_chi2_vals, levels, contour_kw)
    plotting.scan(axes["B"], re_z, im_z, bes_chi2_vals, levels, contour_kw)
    contours = plotting.scan(axes["C"], re_z, im_z, combined_vals, levels, contour_kw)

    for label in ("A", "B", "C"):
        axes[label].add_patch(Circle((0, 0), radius=1, facecolor="none", edgecolor="k"))
        axes[label].set_aspect("equal")

    fig.colorbar(contours, cax=axes["D"])
    # fig.tight_layout()
    fig.savefig(f"charm_bin_{bin_number}.png")


if __name__ == "__main__":
    main()
