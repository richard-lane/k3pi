#!/bin/bash

SOURCE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FLAGS="-shared -Wall -Werror -fPIC -std=c++17"

# CLEO
clang++ k3pi_fitter/lib_time_fit/charm_threshhold/cleo_interface.cpp -o $SOURCE_DIR/cleo.so $FLAGS

# BES
# The BES library requires ROOT
clang++ k3pi_fitter/lib_time_fit/charm_threshhold/bes_interface.cpp -o $SOURCE_DIR/bes.so $FLAGS $(root-config --glibs --cflags --libs)
