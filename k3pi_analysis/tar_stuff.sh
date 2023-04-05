#!/bin/bash
# Tar stuff up

set -ex

# Tar libraries
analysis_tar=analysis.tar.gz
data_tar=data.tar.gz
cuts_tar=cuts.tar.gz
efficiency_tar=efficiency.tar.gz
mass_fit_tar=mass_fit.tar.gz
time_fit_tar=time_fit.tar.gz

if [ ! -f $analysis_tar ]; then
    tar -zcf $analysis_tar k3pi_analysis
fi

if [ ! -f $data_tar ]; then
    # Don't want to tar the dumps!
    tar -zcf $data_tar k3pi-data/lib_data k3pi-data/scripts k3pi-data/create* k3pi-data/branch_names.txt k3pi-data/production_locations
fi

if [ ! -f $cuts_tar ]; then
    tar -zcf $cuts_tar k3pi_signal_cuts
fi

if [ ! -f $efficiency_tar ]; then
    tar -zcf $efficiency_tar k3pi_efficiency
fi

if [ ! -f $mass_fit_tar ]; then
    tar -zcf $mass_fit_tar k3pi_mass_fit
fi

if [ ! -f $time_fit_tar ]; then
    tar -zcf $time_fit_tar k3pi_fitter
fi

# Tar python
python_tar=python.tar.gz
if [ ! -f $python_tar ]; then
    conda pack -n d2k3py -o $python_tar
fi

