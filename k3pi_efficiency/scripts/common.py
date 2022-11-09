"""
Common utilities for running scripts

"""
import argparse


def parser(description: str) -> argparse.ArgumentParser:
    """
    Create an argument parser for all the right arguments

    :param description: description of your script's functionality
    :returns: argument parser object with all arguments added

    """
    retval = argparse.ArgumentParser(description=description)

    retval.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year, so we know which reweighter to open",
    )
    retval.add_argument(
        "decay_type",
        type=str,
        choices={"cf", "dcs", "false"},
        help="CF, DCS or false sign (WS amplitude but RS charges) data",
    )
    retval.add_argument(
        "weighter_type",
        type=str,
        choices={"cf", "dcs"},
        help="Whether to open the reweighter trained on DCS or CF data",
    )
    retval.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="So we know which reweighter to open",
    )
    retval.add_argument(
        "data_k_charge",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to use D->K+ or K- 3pi data (or both)",
    )
    retval.add_argument(
        "weighter_k_charge",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to open the reweighter trained on D0->K+ or K- 3pi",
    )
    retval.add_argument(
        "--fit",
        action="store_true",
        help="So we know which reweighter to open; time fit or hist division",
    )
    retval.add_argument(
        "--cut",
        action="store_true",
        help="So we know which reweighter to open; BDT cut or not",
    )

    return retval


def remove_arg(argparser: argparse.ArgumentParser, arg: str):
    """
    Remove an argument from an ArgumentParser instance

    """
    for action in argparser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            argparser._remove_action(action)
            break

    for action in argparser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return
