#!/bin/bash

set -exu

# Untar the python env that we brought with us
mkdir python
tar -xzf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

# Clone the analysis repo
git clone https://github.com/richard-lane/k3pi.git
cd k3pi

python k3pi_signal_cuts/scripts/optimise/optimise.py --n_procs 8 --n_repeats 2

