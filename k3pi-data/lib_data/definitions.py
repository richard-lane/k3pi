"""
"""
import pathlib
import logging
from typing import List
from glob import glob

D0_MASS_MEV = 1864.84
D0_LIFETIME_PS = 0.4101

DUMP_DIR = pathlib.Path(__file__).resolve().parents[1] / "dumps"


# Column names for momenta
MOMENTUM_SUFFICES = "Px", "Py", "Pz", "E"
MOMENTUM_COLUMNS = [
    *(f"Kplus_{s}" for s in MOMENTUM_SUFFICES),
    *(f"pi1minus_{s}" for s in MOMENTUM_SUFFICES),
    *(f"pi2minus_{s}" for s in MOMENTUM_SUFFICES),
    *(f"pi3plus_{s}" for s in MOMENTUM_SUFFICES),
    *(f"slowpi_{s}" for s in MOMENTUM_SUFFICES),
]

# ROOT file branch prefices
DATA_BRANCH_PREFICES = (
    "Dst_ReFit_D0_Kplus",
    "Dst_ReFit_D0_piplus",
    "Dst_ReFit_D0_piplus_0",
    "Dst_ReFit_D0_piplus_1",
    "Dst_ReFit_piplus",
)

# Particle gun directories
RS_PGUN_SOURCE_DIR = pathlib.Path(
    "/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples/27165071/"
)
WS_PGUN_SOURCE_DIR = pathlib.Path(
    "/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples/27165072/"
)
FALSE_SIGN_SOURCE_DIR = pathlib.Path(
    "/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples/27165077/2018_dw"
)

AMPGEN_DIR = DUMP_DIR / "ampgen"
PGUN_DIR = DUMP_DIR / "pgun"
FALSE_SIGN_DIR = DUMP_DIR / "false_sign_pgun"
MC_DIR = DUMP_DIR / "mc"
DATA_DIR = DUMP_DIR / "data"
UPPERMASS_DIR = DUMP_DIR / "upper_mass"


def data_tree(sign: str) -> str:
    """
    Name of the tree in real data

    """
    assert sign in {"cf", "dcs"}
    return (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "cf"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )


def ampgen_dump(sign: str) -> pathlib.Path:
    """
    Returns the location of the AmpGen dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return AMPGEN_DIR / f"{sign}.pkl"


def mc_files(year: str, magnetisation: str, sign: str) -> List[str]:
    """
    Paths to MC analysis productions on lxplus (or the grid maybe).

    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :param sign: "cf" or "dcs"
    :returns: list of paths as strings

    """
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}
    assert sign in {"cf", "dcs"}

    # File holding locations of productions
    pfn_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "production_locations"
        / "mc"
        / f"{sign}_{year}_{magnetisation}.txt"
    )

    with open(pfn_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def mc_dump(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of the MC dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return MC_DIR / f"{year}_{sign}_{magnetisation}.pkl"


def pgun_dir(sign: str) -> pathlib.Path:
    """
    Returns the location of a directory particle gun dumps

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return PGUN_DIR / sign


def pgun_dump(sign: str, n: int) -> pathlib.Path:
    """
    Returns the location of the n'th particle gun dump

    sign should be "cf" or "dcs"

    """
    return pgun_dir(sign) / f"{n}.pkl"


def false_sign_dump(n: int) -> pathlib.Path:
    """
    Returns the location of the n'th false sign particle gun dump

    """
    return FALSE_SIGN_DIR / f"{n}.pkl"


def data_dir(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of a directory of real data dumps

    :param sign: "cf" or "dcs"
    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: absolute path to the dir used for storing this data

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    return DATA_DIR / f"{year}_{sign}_{magnetisation}"


def data_files(year: str, magnetisation: str) -> List[str]:
    """
    Paths to real data analysis productions on lxplus (or the grid maybe).

    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: list of paths as strings

    """
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    # File holding locations of productions
    pfn_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "production_locations"
        / "real_data"
        / f"{year}_{magnetisation}_dd.txt"
    )

    with open(pfn_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def data_dump(data_file: str, year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Paths to the pickle dump corresponding to a data file.
    Idea is to pass a file returned from `data_files()` as `data_file`

    :param data_file: location of analysis production, e.g. as returned by data_files()
    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"
    :returns: path to the dump location

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    data_file = pathlib.Path(data_file)

    return data_dir(year, sign, magnetisation) / f"{data_file.with_suffix('').name}.pkl"


def uppermass_dir(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of a directory of real data upper mass sideband dumps

    :param sign: "cf" or "dcs"
    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: absolute path to the dir used for storing this data

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    return UPPERMASS_DIR / f"{year}_{sign}_{magnetisation}"


def uppermass_dump(
    data_file: str, year: str, sign: str, magnetisation: str
) -> pathlib.Path:
    """
    Paths to the pickle dump corresponding to a the upper mass sideband of a data file.
    Idea is to pass a file returned from `data_files()` as `data_file`

    :param data_file: location of analysis production, e.g. as returned by data_files()
    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"
    :returns: path to the dump location

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    data_file = pathlib.Path(data_file)

    return (
        uppermass_dir(year, sign, magnetisation)
        / f"{data_file.with_suffix('').name}.pkl"
    )


def pgun_dirs(sign: str) -> List[pathlib.Path]:
    """
    List of filepaths pointing towards particle gun productions on /eos/

    This is just the raw stuff - e.g. not /2018_dw/, or anything

    """
    logging.warning(
        "Use the pgun_filepaths and pgun_dump_fromfile fcns instead",
        exc_info=DeprecationWarning(),
    )

    assert sign in {"dcs", "cf", "false"}
    if sign == "dcs":
        path = WS_PGUN_SOURCE_DIR
    elif sign == "cf":
        path = RS_PGUN_SOURCE_DIR
    else:
        path = FALSE_SIGN_SOURCE_DIR

    return glob(f"{path}[0-9]*/")


def pgun_filepaths(sign: str) -> List[str]:
    """
    Paths to particle gun analysis productions on lxplus (or the grid maybe).

    :param sign: "dcs" or "cf"
    :returns: list of paths as strings

    """
    assert sign in {"dcs", "cf"}

    # File holding locations of productions
    pfn_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "production_locations"
        / "pgun"
        / f"{sign}.txt"
    )

    with open(pfn_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def pgun_dump_fromfile(pgun_file: str, sign: str) -> pathlib.Path:
    """
    Paths to the pickle dump corresponding to a particle gun analysis production file.
    Idea is to pass a file returned from `pgun_filepaths()` as `data_file`

    :param data_file: location of analysis production, e.g. as returned by data_files()
    :param sign: "cf" or "dcs"

    :returns: path to the dump location

    """
    assert sign in {"cf", "dcs"}

    pgun_file = pathlib.Path(pgun_file)

    return pgun_dir(sign) / f"{pgun_file.name}.pkl"
