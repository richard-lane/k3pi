"""
Test a "scan" to see if the plotting fcn is right

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_time_fit import plotting


def _normal(x: float, y: float) -> float:
    """Normal PDF"""
    return 10 * multivariate_normal(
        mean=[0.1, 0.5], cov=[[0.02, 0.0], [0.0, 0.01]]
    ).pdf((x, y))


def main():
    """
    Made up vals of Im/Re Z,
    fill in with a Gaussian,
    plot it as a scan

    """
    n_re, n_im = 100, 101
    allowed_rez = np.linspace(-1, 1, n_re)
    allowed_imz = np.linspace(-1, 1, n_im)

    chi2s = np.ones((n_im, n_re)) * np.inf

    with tqdm(total=n_re * n_im) as pbar:
        for i, re_z in enumerate(allowed_rez):
            for j, im_z in enumerate(allowed_imz):
                chi2s[j, i] = _normal(re_z, im_z)
                pbar.update(1)

    chi2s -= np.min(chi2s)
    chi2s = np.sqrt(chi2s)

    fig, ax = plt.subplots()
    contours = plotting.scan(
        ax,
        allowed_rez,
        allowed_imz,
        chi2s,
        levels=np.arange(10),
    )

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(contours, cax=cbar_ax)
    cbar_ax.set_title(r"$\sigma$")

    plt.show()


if __name__ == "__main__":
    main()
