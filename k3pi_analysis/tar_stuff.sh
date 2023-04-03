#!/bin/bash
# Tar stuff up

set -exu

# Tar libraries
data_tar=lib_data.tar.gz
cuts_tar=lib_cuts.tar.gz
efficiency_tar=lib_efficiency.tar.gz
mass_fit_tar=mass_fit.tar.gz
time_fit_tar=time_fit.tar.gz

if [ ! -f $data_tar ]; then
    tar -zcf $data_tar k3pi-data/lib_data
fi

if [ ! -f $cuts_tar ]; then
    tar -zcf $cuts_tar k3pi_signal_cuts/lib_cuts
fi

if [ ! -f $efficiency_tar ]; then
    tar -zcf $efficiency_tar k3pi_efficiency/lib_efficiency
fi

if [ ! -f $mass_fit_tar ]; then
    tar -zcf $mass_fit_tar k3pi_mass_fit/libFit
fi

if [ ! -f $time_fit_tar ]; then
    tar -zcf $time_fit_tar k3pi_fitter/lib_time_fit
fi

# Tar python
python_tar=python.tar.gz
if [ ! -f $python_tar ]; then
    conda pack -n d2k3py -o $python_tar
fi

