#!/bin/bash
# This is the script that gets run on the grid
# Unzips things, sets of the analysis script

set -ex

# Untar the python env that we brought with us
mkdir python
tar -xzf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

# Clone the analysis repo
git clone https://github.com/richard-lane/k3pi.git

# cd into the right dir
cd k3pi

# Run the analysis
./k3pi_analysis/analysis.sh

# Allow errors now that the analysis has actually run
set +e

# Move all the output stuff into a special dir
mkdir out_stuff/
mv *.png out_stuff/
mv *.svg out_stuff/
mv *.txt out_stuff
mv k3pi_mass_fit/*.txt out_stuff/
mv raw_fits out_stuff/
mv bdt_fits out_stuff/
mv eff_fits out_stuff/
mv alt_bkg_eff_fits out_stuff/

# Move the output dir into the home dir
mv out_stuff/ ..

# cd into home dir
cd ..

# Tar the output stuff up
tar -czf out_files_${CONDOR_JOB_ID}.tar.gz out_stuff/

# debug
find . -maxdepth 1 -type f

