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

# Add phsp information to the real data dfs
python k3pi-data/scripts/add_phsp_bins.py 2018 dcs magdown &
pids[2]=$!
python k3pi-data/scripts/add_phsp_bins.py 2018 cf magdown &
pids[3]=$!

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
python k3pi_signal_cuts/create_classifier.py 2018 dcs magdown &
pids[0]=$!

# Perform mass fits without BDT cut
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 0 &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 1 &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 2 &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 3 &
pids[4]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown &
pids[5]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot BDT stuff
# These are quick so no need to paralellise
python k3pi_signal_cuts/scripts/plot_roc.py
python k3pi_signal_cuts/scripts/plot_cuts.py
python k3pi_signal_cuts/scripts/plot_data_cuts.py

# Perform mass fits with BDT cut
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 0 --bdt_cut &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 1 --bdt_cut &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 2 --bdt_cut &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown 3 --bdt_cut &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py 2018 magdown --bdt_cut &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot yields
python k3pi_mass_fit/scripts/plot_yield_from_file.py 2018 magdown 0 1 2 3 --integrated
python k3pi_mass_fit/scripts/plot_yield_from_file.py 2018 magdown 0 1 2 3 --integrated --bdt_cut

