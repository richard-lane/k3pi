#/bin/bash
# Tar stuff up, submit a job to the grid

set -exu

./k3pi_analysis/tar_stuff.sh

condor_submit k3pi_analysis/submit.sub

condor_q
