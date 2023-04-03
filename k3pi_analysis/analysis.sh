#/bin/bash

# Run the whole analysis
# Gets the data, trains "BDT-cut" classifier, trains efficiency reweighter,
# performs mass fits and a fit to the yield
# Also does a few toy studies/tests/etc. to verify that things are working

# Treat unset variables as error
set -u

# Exit immediately if any script exits non-zero
set -e

# Print commands and their arguments as they are executed
set -x

# Create the right dataframes
# Need to point the AmpGen scripts at the ROOT files that
# were generated with AmpGen
python k3pi-data/create_ampgen.py ./ws_D02piKpipi.root dcs &
pids[0]=$!
python k3pi-data/create_ampgen.py ./rs_Dbar02piKpipi.root cf &
pids[1]=$!

# Only create a few pgun dfs for speed
python k3pi-data/create_pgun.py 2018 dcs magdown -n 8 -v &
pids[2]=$!
python k3pi-data/create_pgun.py 2018 cf magdown -n 8 -v &
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

# Only create a few real dfs for speed
# Don't run these in parallel since they spawn their own processes
python k3pi-data/create_real.py -n 12 2018 cf magdown --n_procs 6
python k3pi-data/create_real.py -n 12 2018 dcs magdown --n_procs 6

# Plot projections of data
python k3pi-data/scripts/plot_parameterisation.py
