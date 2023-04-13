#!/bin/bash

set -ex

# Untar the python env that we brought with us
mkdir python
tar -xzf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

# Clone the analysis repo
git clone https://github.com/richard-lane/k3pi.git
cd k3pi

# Run the optimisation
python k3pi_signal_cuts/scripts/optimise/optimise.py 8 2

# Debug
find . -type f

# Move the output stuff back to the home dir
mv ${CONDOR_JOB_ID}_bdt_opt.png ../
mv ${CONDOR_JOB_ID}_bdt_opt.pkl ../
cd ..

# Debug
find . -maxdepth 1

