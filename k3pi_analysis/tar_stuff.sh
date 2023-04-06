#!/bin/bash
# Tar stuff up
# Just the python environment

set -ex

# Tar python
python_tar=python.tar.gz
if [ ! -f $python_tar ]; then
    conda pack -n d2k3py -o $python_tar
fi

