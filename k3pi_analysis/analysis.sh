#/bin/bash

# These need to be set as env vars
set -u
echo "YEAR: $YEAR MAG: $MAG"
set +u

# Run the whole analysis
# Gets the data, trains "BDT-cut" classifier, trains efficiency reweighter,
# performs mass fits and a fit to the yield
# Also does a few toy studies/tests/etc. to verify that things are working

# In order, we first train get the necessary data + train the BDT
# Then begin training efficiency weighters, because this takes a long time
# While this is happening, we create the real data dataframes
# Then do the mass fit + time plots

# Exit immediately if any script exits non-zero
# Print commands and their arguments as they are executed
set -ex

# Build the shared libraries that we'll need later
./k3pi_efficiency/lib_efficiency/amplitude_models/build.sh &
pids[0]=$!
./k3pi_fitter/lib_time_fit/charm_threshhold/build.sh &
pids[1]=$!

# Create the right dataframes
# First create uppermass + MC
python k3pi-data/create_uppermass.py $YEAR dcs $MAG -n 100 --n_procs 6 &
pids[0]=$!

python k3pi-data/create_mc.py $YEAR dcs $MAG &
pids[1]=$!
python k3pi-data/create_mc.py $YEAR cf $MAG &
pids[2]=$!

wait ${pids[1]}
wait ${pids[2]}

# Start creating particle gun dfs
python k3pi-data/create_pgun.py $YEAR dcs $MAG -v &
pids[1]=$!
python k3pi-data/create_pgun.py $YEAR cf $MAG -v &
pids[2]=$!

wait ${pids[1]}
wait ${pids[2]}

# Start creating ampgen dfs
# Need to point the AmpGen scripts at the ROOT files that
# were generated with AmpGen
# when using HTcondor, these were brought over and live in the worker node's home dir
python k3pi-data/create_ampgen.py ../ws_D02piKpipi.root dcs &
pids[1]=$!
python k3pi-data/create_ampgen.py ../rs_Dbar02piKpipi.root cf &
pids[2]=$!

# Wait for the dfs to be made
wait ${pids[0]}
wait ${pids[1]}
wait ${pids[2]}
unset pids

# Train BDT
python k3pi_signal_cuts/create_classifier.py $YEAR dcs $MAG &
pids[0]=$!

# Wait for ampgen + particle gun dfs + the bdt to finish
wait ${pids[0]}
wait ${pids[3]}
wait ${pids[4]}
wait ${pids[5]}
wait ${pids[6]}
unset pids

# Start training efficiency reweighters; they are extremely slow
python k3pi_efficiency/create_reweighter.py --cut cf $YEAR $MAG both &
cf_eff_pid=$!
python k3pi_efficiency/create_reweighter.py --cut dcs $YEAR $MAG both &
dcs_eff_pid=$!

# Create real data dataframes now as well; this is also slow
python k3pi-data/create_real.py $YEAR cf $MAG --n_procs 3 &
cf_data_pid=$!
python k3pi-data/create_real.py $YEAR dcs $MAG --n_procs 3 &
dcs_data_pid=$!

# This should be most of the analysis, in terms of time
wait $cf_eff_pid
wait $dcs_eff_pid
wait $cf_data_pid
wait $dcs_data_pid

unset cf_eff_pid
unset dcs_eff_pid
unset cf_data_pid
unset dcs_data_pid

# Plot efficiency validation plots
python k3pi_efficiency/scripts/plot_projection.py $YEAR cf cf $MAG both both --cut &
pids[0]=$!
python k3pi_efficiency/scripts/plot_projection.py $YEAR dcs dcs $MAG both both --cut &
pids[1]=$!
python k3pi_efficiency/scripts/plot_z_scatter.py $YEAR cf cf $MAG both both --cut &
pids[2]=$!
python k3pi_efficiency/scripts/plot_z_scatter.py $YEAR dcs dcs $MAG both both --cut &
pids[3]=$!
python k3pi_efficiency/scripts/plot_time_ratio.py $YEAR $MAG both both --cut &
pids[4]=$!

# Create false sign dumps
python k3pi-data/create_false_sign_pgun.py &
pids[5]=$!

for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# False sign validation plots
python k3pi_efficiency/scripts/plot_projection.py $YEAR 'false' cf $MAG both both --cut &
pids[0]=$!
python k3pi_efficiency/scripts/plot_projection.py $YEAR 'false' dcs $MAG both both --cut &
pids[1]=$!
python k3pi_efficiency/scripts/plot_z_scatter.py $YEAR 'false' cf $MAG both both --cut &
pids[2]=$!
python k3pi_efficiency/scripts/plot_z_scatter.py $YEAR 'false' dcs $MAG both both --cut &
pids[3]=$!

for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Add phsp information to the real data dfs
python k3pi-data/scripts/add_phsp_bins.py $YEAR dcs $MAG &
pids[0]=$!
python k3pi-data/scripts/add_phsp_bins.py $YEAR cf $MAG &
pids[1]=$!

# Do ip fits in first time bin to fix prompt shape
python k3pi-data/scripts/ipchi2_fit_low_t.py $YEAR dcs $MAG &
pids[2]=$!
python k3pi-data/scripts/ipchi2_fit_low_t.py $YEAR cf $MAG &
pids[3]=$!

# Once we have the real data stuff we're ready to do the fits
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Do IP fits to get secondary fractions
python k3pi-data/scripts/data_ipchi2_fit.py $YEAR dcs $MAG &
pids[0]=$!
python k3pi-data/scripts/data_ipchi2_fit.py $YEAR cf $MAG
pids[1]=$!

# Plot BDT stuff
python k3pi_signal_cuts/scripts/plot_roc.py $YEAR dcs $MAG &
pids[2]=$!
python k3pi_signal_cuts/scripts/plot_cuts.py $YEAR dcs $MAG &
pids[3]=$!
python k3pi_signal_cuts/scripts/plot_data_cuts.py $YEAR dcs $MAG &
pids[4]=$!
python k3pi_signal_cuts/scripts/plot_signal_significance.py $YEAR dcs $MAG &
pids[5]=$!

# Plot projections of data
python k3pi-data/scripts/plot_parameterisation.py $YEAR $MAG &
pids[6]=$!

for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# mass fits
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 0 --bdt_cut --efficiency &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 1 --bdt_cut --efficiency &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 2 --bdt_cut --efficiency &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 3 --bdt_cut --efficiency &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG --bdt_cut --efficiency &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# mass fits without bdt or efficiency, just becuase
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 0 &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 1 &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 2 &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 3 &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot yields for all phsp bins + also for bin integrated
python k3pi_mass_fit/scripts/plot_yield_from_file.py $YEAR $MAG 0 1 2 3 --integrated --bdt_cut --efficiency

# Plot the fits using the no mixing/mixing hypotheses
python k3pi_fitter/scripts/mixing_fit_from_file.py --bdt_cut --efficiency $YEAR $MAG --sec_correction --misid_correction

# Plot Z scans (LHCb only)
python k3pi_fitter/scripts/lhcb_fit_from_file.py $YEAR $MAG -1 --bdt_cut --efficiency --sec_correction --misid_correction

python k3pi_fitter/scripts/lhcb_fit_from_file.py $YEAR $MAG 0 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/lhcb_fit_from_file.py $YEAR $MAG 1 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/lhcb_fit_from_file.py $YEAR $MAG 2 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/lhcb_fit_from_file.py $YEAR $MAG 3 --bdt_cut --efficiency --sec_correction --misid_correction

# Plot fits with charm constraint
# These have to be phase space binned, since the BES constraint is binned
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 0 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 1 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 2 --bdt_cut --efficiency --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 3 --bdt_cut --efficiency --sec_correction --misid_correction

#### Systematic stuff
# Charm fits without secondary correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 0 --bdt_cut --efficiency --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 1 --bdt_cut --efficiency --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 2 --bdt_cut --efficiency --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 3 --bdt_cut --efficiency --misid_correction

# Charm fits without double misid correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 0 --bdt_cut --efficiency --sec_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 1 --bdt_cut --efficiency --sec_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 2 --bdt_cut --efficiency --sec_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 3 --bdt_cut --efficiency --sec_correction

# Perform mass fits with no efficiency (but with BDT cut)
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 0 --bdt_cut &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 1 --bdt_cut &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 2 --bdt_cut &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 3 --bdt_cut &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG --bdt_cut &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Fits without efficiency correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 0 --bdt_cut --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 1 --bdt_cut --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 2 --bdt_cut --sec_correction --misid_correction
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 3 --bdt_cut --sec_correction --misid_correction

# Create bkg pdf dumps
python k3pi_mass_fit/scripts/create_bkg.py $YEAR $MAG dcs --n_repeats 500 --bdt_cut &
pids[0]=$!
python k3pi_mass_fit/scripts/create_bkg.py $YEAR $MAG cf --n_repeats 250 --bdt_cut &
pids[1]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Plot them, and the PDF
python k3pi_mass_fit/scripts/plot_bkg.py $YEAR $MAG --bdt_cut

# Perform mass fits with alternate background model
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 0 --bdt_cut --alt_bkg --efficiency &
pids[0]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 1 --bdt_cut --alt_bkg --efficiency &
pids[1]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 2 --bdt_cut --alt_bkg --efficiency &
pids[2]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG 3 --bdt_cut --alt_bkg --efficiency &
pids[3]=$!
python k3pi_mass_fit/scripts/data_fits.py $YEAR $MAG --bdt_cut --alt_bkg --efficiency &
pids[4]=$!
for pid in ${pids[*]}; do
    wait $pid
done
unset pids

# Do charm scans with alt bkg model
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 0 --bdt_cut --efficiency --sec_correction --misid_correction --alt_bkg
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 1 --bdt_cut --efficiency --sec_correction --misid_correction --alt_bkg
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 2 --bdt_cut --efficiency --sec_correction --misid_correction --alt_bkg
python k3pi_fitter/scripts/fit_from_file.py $YEAR $MAG 3 --bdt_cut --efficiency --sec_correction --misid_correction --alt_bkg

