#!/bin/bash
# Do the rD mixing study with all the different conditions

# cwd
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo -ne 'all:\t\t'
SCRIPT=$SCRIPT_DIR/mixing_fit_from_file.py
python $SCRIPT all all --bdt_cut --efficiency --sec_correction --misid_correction

echo -ne 'efficiency:\t'
python $SCRIPT all all --bdt_cut --sec_correction --misid_correction

echo -ne 'alt bkg:\t'
python $SCRIPT all all --bdt_cut --efficiency --alt_bkg --sec_correction --misid_correction

echo -ne 'sec corr:\t'
python $SCRIPT all all --bdt_cut --efficiency --misid_correction

echo -ne 'misid corr:\t'
python $SCRIPT all all --bdt_cut --efficiency --sec_correction
