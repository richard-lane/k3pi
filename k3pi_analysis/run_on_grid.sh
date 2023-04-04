#!/bin/bash
# This is the script that gets run on the grid
# Unzips things, sets of the analysis script

set -exu

# Untar the libraries that we brought with us
tar -xzf lib_data.tar.gz
tar -xzf lib_cuts.tar.gz
tar -xzf lib_efficiency.tar.gz
tar -xzf mass_fit.tar.gz
tar -xzf time_fit.tar.gz

# Untar the python env that we brought with us
mkdir python
tar -xzf python.tar.gz --directory=python

# Source the python executable I want to use
source python/bin/activate

# Move the scripts to the right directories, since it uses relative imports
mkdir k3pi_analysis/
mv analysis.sh k3pi_analysis/

mkdir -p k3pi-data/scripts/
mv plot_parameterisation.py k3pi-data/scripts/

# Run the analysis
./k3pi_analysis/analysis.sh

