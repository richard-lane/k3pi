#!/bin/bash

set -e

# Tar libraries
data_tar=lib_data.tar.gz
cuts_tar=lib_cuts.tar.gz
if [ ! -f $data_tar ]; then
    tar -zcvf $data_tar k3pi-data/lib_data
fi

if [ ! -f $cuts_tar ]; then
    tar -zcvf $cuts_tar k3pi_signal_cuts/lib_cuts
fi

# Tar python
python_tar=python.tar.gz
if [ ! -f $python_tar ]; then
    conda pack -n d2k3py -o $python_tar
fi
