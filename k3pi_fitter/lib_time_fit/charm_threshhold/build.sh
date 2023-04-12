#!/bin/bash

# Exit immediately on error
set -exu

SOURCE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FLAGS="-shared -Wall -Werror -fPIC -std=c++17 -static-libstdc++"
GPP_CXX=/cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0-afc57/x86_64-centos7/bin/g++

# CLEO
$GPP_CXX $SOURCE_DIR/cleo_interface.cpp -o $SOURCE_DIR/cleo.so $FLAGS

# BES
# I do something funny here
# Create a c++ source file that contains the BESIII covariance matrix
echo "generating source..."
python $SOURCE_DIR/compile_covmatrix.py

# Then compile it so we don't have to read the ROOT file every time
# The BES library requires ROOT
echo "building cov matrix shared lib..."
BES_LIB=$SOURCE_DIR/libBes.so
$GPP_CXX $SOURCE_DIR/bes_interface.cpp $SOURCE_DIR/besIII_covmat.cpp -o $BES_LIB \
    $FLAGS $(root-config --glibs --cflags --libs)

# Also compile a test
echo "building test..."
TEST_DIR=$SOURCE_DIR/../../../test
INCLUDE_LOC=$TEST_DIR/../
$GPP_CXX $TEST_DIR/test_bes_likelihood.cpp -o $TEST_DIR/bes_covariance_test.exe -Wall -Werror -std=c++17 -L$SOURCE_DIR $BES_LIB\
    -Wl,-rpath,$SOURCE_DIR

# Run the test
echo "running test..."
$TEST_DIR/bes_covariance_test.exe

echo "done!"
