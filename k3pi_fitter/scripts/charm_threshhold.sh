#!/bin/bash

set -eux

python k3pi_fitter/scripts/charm_threshhold.py 0 &
pids[0]=$!
python k3pi_fitter/scripts/charm_threshhold.py 1 &
pids[1]=$!
python k3pi_fitter/scripts/charm_threshhold.py 2 &
pids[2]=$!
python k3pi_fitter/scripts/charm_threshhold.py 3 &
pids[3]=$!

for pid in ${pids[*]}; do
    wait $pid
done

convert -append charm_bin_0.png charm_bin_1.png charm_bin_2.png charm_bin_3.png charm_threshhold_chi2s.png

