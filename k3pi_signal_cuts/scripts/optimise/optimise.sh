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

# Create dataframes
YEAR=2018
MAG=magdown
python k3pi-data/create_uppermass.py $YEAR dcs $MAG -n 36 &
pids[0]=$!
python k3pi-data/create_mc.py $YEAR dcs $MAG &
pids[1]=$!

for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Run the optimisation
python k3pi_signal_cuts/scripts/optimise/optimise.py 8 25

# Debug
find . -type f

# Move the output stuff back to the home dir
mv ${CONDOR_JOB_ID}_bdt_opt.png ../
mv ${CONDOR_JOB_ID}_bdt_opt.pkl ../
cd ..

# Debug
find . -maxdepth 1

