"""
Definitions for the amplitude models

"""
from numpy import pi, exp

# Scaling + rotation for amplitudes so that dcs/cf amplitude ratio ~ 0.055 and relative strong phase ~ 0
DCS_OFFSET: complex = 0.0601387 * exp((0 + 1j) * 1.04827 * pi / 180.0)
