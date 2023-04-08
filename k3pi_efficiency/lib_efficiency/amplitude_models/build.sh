#!/bin/bash

set -e

# Get the right directory for putting the libraries in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# I had some problems with the wrong version of g++ so just hard code it here
# It should work on lxplus and probably grid jobs
GPP_CXX=/cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0-afc57/x86_64-centos7/bin/g++
$GPP_CXX -fPIC -Ofast -rdynamic --std=c++14 -march=native -shared $DIR/cf_wrapper.cpp -I $DIR -o $DIR/cf_wrapper.so
$GPP_CXX -fPIC -Ofast -rdynamic --std=c++14 -march=native -shared $DIR/dcs_wrapper.cpp -I $DIR -o $DIR/dcs_wrapper.so

# Otherwise use this one
# g++ -fPIC -Ofast -rdynamic --std=c++14 -march=native -shared $DIR/cf_wrapper.cpp -I $DIR -o $DIR/cf_wrapper.so
# g++ -fPIC -Ofast -rdynamic --std=c++14 -march=native -shared $DIR/dcs_wrapper.cpp -I $DIR -o $DIR/dcs_wrapper.so
