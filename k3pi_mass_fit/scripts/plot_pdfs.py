import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

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

    sig_frac = 0.2
    pdf = pdfs.fractional_pdf(
        centres, sig_frac, centre, width_l, width_r, alpha_l, alpha_r, beta, a, b
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].plot(centres, sig)
    ax[0, 0].plot(centres, bkg)

    ax[0, 1].plot(centres, pdf)
    ax[0, 1].set_title(f"Signal fraction = {sig_frac}")

    sig_cum = pdfs.signal_cdf(
        centres,
        centre,
        width_l,
        width_r,
        alpha_l,
        alpha_r,
        beta,
    )
    bkg_cum = pdfs.bkg_cdf(centres, a, b)
    cdf = pdfs.cdf(
        centres, sig_frac, centre, width_l, width_r, alpha_l, alpha_r, beta, a, b
    )

    ax[1, 0].plot(centres, sig_cum)
    ax[1, 0].plot(centres, bkg_cum)

    ax[1, 1].plot(centres, cdf)

    plt.show()


if __name__ == "__main__":
    main()
