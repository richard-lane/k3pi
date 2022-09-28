#!/bin/bash

# Untar the libraries that we brought with us
tar -xzf lib_data.tar.gz
tar -xzf lib_efficiency.tar.gz

# I dont know why this has happened but lib_data is inside the k3pi-data dir
mv k3pi-data/lib_data .
rmdir k3pi-data

# Untar the python env that we brought with us
tar -xzf python.tar.gz

# Source the python executable I want to use
# NB a good solution would use Docker, which is maybe
# something I'll try to do
source miniconda3/bin/activate

python optimise.py dcs k_minus
