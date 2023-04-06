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

# Move the output stuff into the home dir
ls  # debug
mv data_param_?s.png ..

# cd into home dir for some reason
cd ..

# debug
find . -maxdepth 1 -type f

