#!/bin/bash
# Do the rD mixing study with all the different conditions

# cwd
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT=$SCRIPT_DIR/fit_from_file.py
echo -ne 'all:\t\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --sec_correction --misid_correction --quiet 2>/dev/null

echo -ne 'efficiency:\t'
python $SCRIPT all all $1 --bdt_cut --sec_correction --misid_correction --quiet 2>/dev/null

echo -ne 'alt bkg:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --alt_bkg --sec_correction --misid_correction --quiet 2>/dev/null


echo -ne 'sec corr:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --misid_correction --quiet 2>/dev/null


echo -ne 'misid corr:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --sec_correction --quiet 2>/dev/null

