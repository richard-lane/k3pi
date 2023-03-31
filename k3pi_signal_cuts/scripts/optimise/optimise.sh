#!/bin/bash

set -e

# Untar the libraries that we brought with us
tar -xzf lib_data.tar.gz
tar -xzf lib_cuts.tar.gz

# Untar the python env that we brought with us
mkdir python
tar -xzf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

# Move the script to the right directory, since it uses relative imports
mkdir -p k3pi_signal_cuts/scripts/optimise/
mv optimise.py k3pi_signal_cuts/scripts/optimise

# debug
find . -type f -name '*.py'

python k3pi_signal_cuts/scripts/optimise/optimise.py

