"""
Mixing parameters from the PDG

"""


def d_mass() -> float:
    """ D mass from PDG in MeV """
    return 1864.84


def d_width() -> float:
    """ D width from the PDG in MeV^-1 """
    d_lifetime_s = 410.3  # x10^-15
    sec_to_inv_mev = 1 / 6.58  # x10^22

    return 1 / (d_lifetime_s * sec_to_inv_mev * 10 ** 7)


def mixing_x() -> float:
    """ mixing x from pdg, dimensionless """
    return 410.3 * 0.997 * 10 ** -5


def mixing_y() -> float:
    """ mixing y from pdg, dimensionless """
    return 1.23 * 0.01 / 2
