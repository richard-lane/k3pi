#!/bin/bash

SOURCE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# TODO make this put the library in the right place (this dir)
clang++ k3pi_fitter/lib_time_fit/charm_threshhold/cleo_interface.cpp -shared -o $SOURCE_DIR/cleo.so -Wall -Werror -fPIC
