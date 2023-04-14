"""
Useful definitions

"""
import numpy as np

TIME_BINS = np.array(
    (-1.0, 0.0, 0.94, 1.185, 1.40, 1.62, 1.85, 2.13, 2.45, 2.87, 3.5, 8.0, 19.0)
)

# From https://hflav-eos.web.cern.ch/hflav-eos/charm/ICHEP22/figures/MINUIT_ICHEP22.pdf
CHARM_X = 0.0043069
CHARM_X_ERR = 0.0013427

CHARM_Y = 0.0064743
CHARM_Y_ERR = 0.0002459

CHARM_XY_CORRELATION = -0.301
