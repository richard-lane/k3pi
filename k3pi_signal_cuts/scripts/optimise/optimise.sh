#!/bin/bash

set -e

# Untar the libraries that we brought with us
tar -xzvf lib_data.tar.gz
tar -xzvf lib_cuts.tar.gz

# Untar the python env that we brought with us
mkdir python
tar -xzvf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

python optimise.py

