#/bin/bash

# Run the whole analysis
# Gets the data, trains "BDT-cut" classifier, trains efficiency reweighter,
# performs mass fits and a fit to the yield
# Also does a few toy studies/tests/etc. to verify that things are working

# Print commands and their arguments as they are executed
# Exit immediately if any script exits non-zero
set -ex

# Create the right dataframes
# Need to point the AmpGen scripts at the ROOT files that
# were generated with AmpGen
# These were brought over and live in the worker node's home dir
python k3pi-data/create_ampgen.py ../ws_D02piKpipi.root dcs &
pids[0]=$!
python k3pi-data/create_ampgen.py ../rs_Dbar02piKpipi.root cf &
pids[1]=$!

# Only create a few pgun dfs for speed
python k3pi-data/create_pgun.py 2018 dcs magdown -n 24 -v &
pids[2]=$!
python k3pi-data/create_pgun.py 2018 cf magdown -n 24 -v &
pids[3]=$!

python k3pi-data/create_mc.py 2018 dcs magdown &
pids[4]=$!
python k3pi-data/create_mc.py 2018 cf magdown &
pids[5]=$!

# Wait for these to all be done
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Create some uppermass dfs
python k3pi-data/create_uppermass.py 2018 dcs magdown -n 24 &
pids[0]=$!
python k3pi-data/create_uppermass.py 2018 cf magdown -n 24 &
pids[1]=$!

for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Only create a few real dfs for speed
# Don't run these in parallel since they spawn their own processes
python k3pi-data/create_real.py -n 18 2018 cf magdown --n_procs 6
python k3pi-data/create_real.py -n 18 2018 dcs magdown --n_procs 6

# Plot projections of data
python k3pi-data/scripts/plot_parameterisation.py

# Train BDT
python k3pi_signal_cuts/create_classifier.py 2018 dcs magdown

# Plot BDT stuff
python k3pi_signal_cuts/scripts/plot_roc.py
python k3pi_signal_cuts/scripts/plot_cuts.py
python k3pi_signal_cuts/scripts/plot_data_cuts.py

