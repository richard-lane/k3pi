#/bin/bash

YEAR=2018
MAG=magdown

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
python k3pi-data/create_pgun.py $YEAR dcs $mag -v &
pids[2]=$!
python k3pi-data/create_pgun.py $YEAR cf $mag -v &
pids[3]=$!

# MC dataframes are quick so just stick them in here
python k3pi-data/create_mc.py $YEAR dcs $mag &
pids[4]=$!
python k3pi-data/create_mc.py $YEAR cf $mag &
pids[5]=$!

# Wait for ampgen and particle dfs to be made
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Start training efficiency reweighters now because they are extremely slow
python k3pi_efficiency/create_reweighter.py --cut cf $YEAR $mag both &
pids[0]=$!
python k3pi_efficiency/create_reweighter.py --cut dcs $YEAR $mag both &
pids[1]=$!

# Only create a few real dfs for speed
python k3pi-data/create_real.py -n 48 $YEAR cf $mag --n_procs 3 &
pids[2]=$!
python k3pi-data/create_real.py -n 48 $YEAR dcs $mag --n_procs 3 &
pids[3]=$!

# This should be most of the analysis, in terms of time
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Build amplitude model libraries
./k3pi_efficiency/lib_efficiency/amplitude_models/build.sh

# Create some uppermass dfs
python k3pi-data/create_uppermass.py $YEAR dcs $mag -n 48 &
pids[0]=$!
python k3pi-data/create_uppermass.py $YEAR cf $mag -n 48 &
pids[1]=$!

# Add phsp information to the real data dfs
python k3pi-data/scripts/add_phsp_bins.py $YEAR dcs $mag &
pids[2]=$!
python k3pi-data/scripts/add_phsp_bins.py $YEAR cf $mag &
pids[3]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot projections of data
python k3pi-data/scripts/plot_parameterisation.py

# Train BDT
python k3pi_signal_cuts/create_classifier.py $YEAR dcs $mag &
pids[0]=$!

# Perform mass fits without BDT cut
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 0 &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 1 &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 2 &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 3 &
pids[4]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag &
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
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 0 --bdt_cut &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 1 --bdt_cut &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 2 --bdt_cut &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 3 --bdt_cut &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag --bdt_cut &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Perform mass fits with BDT + efficiency
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 0 --bdt_cut --efficiency &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 1 --bdt_cut --efficiency &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 2 --bdt_cut --efficiency &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag 3 --bdt_cut --efficiency &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $mag --bdt_cut --efficiency &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot yields
python k3pi_mass_fit/scripts/plot_yield_from_file.py $YEAR $mag 0 1 2 3 --integrated
python k3pi_mass_fit/scripts/plot_yield_from_file.py $YEAR $mag 0 1 2 3 --integrated --bdt_cut
python k3pi_mass_fit/scripts/plot_yield_from_file.py $YEAR $mag 0 1 2 3 --integrated --bdt_cut --efficiency

# Plot fits
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 0 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 1 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 2 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 3 --bdt_cut

python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 0 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 1 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 2 --bdt_cut
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 3 --bdt_cut

python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 0 --bdt_cut --efficiency
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 1 --bdt_cut --efficiency
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 2 --bdt_cut --efficiency
python k3pi_fitter/scripts/fit_from_file.py $YEAR $mag 3 --bdt_cut --efficiency

