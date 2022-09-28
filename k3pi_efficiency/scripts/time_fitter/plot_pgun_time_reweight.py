"""
Plot the result of reweighting the particle gun data - should look like AmpGen

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "k3pi-data"))

from lib_efficiency import reweighter
from lib_data import get


def _times(dataframe):
    """
    Train + test times

    """
    train = dataframe["train"]
    times = dataframe["time"]

    return times[train], times[~train]


def main(sign: str):
    """
    Create plots showing the time reweighting with and without using the fitter

    """
    pgun_train, pgun_test = _times(get.particle_gun(sign, show_progress=True))
    ampgen_train, ampgen_test = _times(get.ampgen(sign))

    # Don't plot below the minimum time
    min_t = 0.4
    reweighters = [
        reweighter.TimeWeighter(min_t, fit=False),
        reweighter.TimeWeighter(min_t, fit=True),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    hist_kw = {"density": True, "bins": np.linspace(0.0, 8, 100)}

    for weighter, axis, title in zip(reweighters, ax, ("Histogram", "fit")):
        weighter.fit(pgun_train, ampgen_train)
        axis.hist(
            pgun_test[pgun_test > min_t], **hist_kw, label="before", histtype="step"
        )
        axis.hist(
            ampgen_test[ampgen_test > min_t],
            **hist_kw,
            label="AmpGen",
            histtype="step",
        )
        axis.hist(
            pgun_test,
            **hist_kw,
            label="after",
            weights=weighter.correct_efficiency(pgun_test),
            alpha=0.5,
        )

        axis.set_title(title)

    pts = np.linspace(0, 8)
    pdf = reweighters[1].fitter._fitted_pdf(pts)
    plt.plot(pts, pdf)
    plt.show()

    ax[1].legend()
    fig.suptitle(f"Particle Gun {sign} Reweighting")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})

    args = parser.parse_args()
    main(args.sign)
