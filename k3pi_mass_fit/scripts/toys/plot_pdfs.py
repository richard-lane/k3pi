import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2]))

from libFit import pdfs


def main():
    centres = np.linspace(*pdfs.domain(), 1000)
    mean, *sig_params = pdfs.signal_defaults(time_bin=5)
    _, *bkg_params = pdfs.background_defaults("RS")

    centre, width_l, alpha_l, beta, a, b = pdfs.defaults("RS", 5)
    width_r = 1.5 * width_l
    alpha_r = 1.5 * alpha_l

    sig = pdfs.normalised_signal(
        centres,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
    )
    bkg = pdfs.normalised_bkg(centres, a, b)

    n_sig = 0.2
    n_bkg = 1 - n_sig
    pdf = pdfs.model(
        centres,
        n_sig,
        n_bkg,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
        a,
        b,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(centres, sig)
    ax[0].plot(centres, bkg)

    ax[1].plot(centres, pdf)
    ax[1].set_title(f"Signal fraction = {n_sig}")
    plt.show()


if __name__ == "__main__":
    main()
