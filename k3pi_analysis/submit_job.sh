#/bin/bash
# Tar stuff up, submit a job to the grid

set -exu

if (( $# != 2 )); then
    >&2 echo "Pass year and mag"
    exit 2
fi

./k3pi_analysis/tar_stuff.sh

SCRIPT="k3pi_analysis/submit_${1}_${2}.sub"
echo "submitting $SCRIPT"

condor_submit "k3pi_analysis/submit.sub"

condor_q
