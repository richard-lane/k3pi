"""
Calculate the systematic error on a parameter from the
systematic pull error scaling

"""
import argparse
from math import sqrt


def main(*, err1: float, err2: float):
    """
    Combine the errors

    """
    diff = abs(err1 - err2)
    print(sqrt(diff * (err1 + err2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum errors")
    parser.add_argument("err1", type=float, help="One of the errors")
    parser.add_argument("err2", type=float, help="The other one of the errors")

    main(**vars(parser.parse_args()))
