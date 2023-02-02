"""
Get the reweighter

"""
import pickle
from . import efficiency_definitions
from .reweighter import EfficiencyWeighter


def reweighter_dump(
    year: str,
    sign: str,
    magnetisation: str,
    k_sign: str,
    fit: bool,
    cut: bool,
    verbose: bool = False,
) -> EfficiencyWeighter:
    """
    Get the right reweighter from a pickle dump

    """
    # Find the right reweighter to unpickle
    reweighter_path = efficiency_definitions.reweighter_path(
        year,
        sign,
        magnetisation,
        k_sign,
        fit,
        cut,
    )

    # Open the reweighter
    if verbose:
        print(f"Opening reweighter at {reweighter_path}")
    with open(reweighter_path, "rb") as f:
        return pickle.load(f)


def ampgen_reweighter_dump(sign: str, verbose: bool = False) -> EfficiencyWeighter:
    """
    Get the right reweighter from a pickle dump

    """
    # Find the right reweighter to unpickle
    reweighter_path = efficiency_definitions.ampgen_reweighter_path(sign)

    # Open the reweighter
    if verbose:
        print(f"Opening reweighter at {reweighter_path}")
    with open(reweighter_path, "rb") as f:
        return pickle.load(f)
