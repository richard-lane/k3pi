#!/bin/bash

set -e

./k3pi_analysis/tar_stuff.sh

condor_submit k3pi_signal_cuts/scripts/optimise/submit.sub

condor_q

