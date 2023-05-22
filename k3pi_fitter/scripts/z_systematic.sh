#!/bin/bash
# Do the rD mixing study with all the different conditions

set -e

if [ "$#" -ne 1 ]; then
    echo "Provide one phsp bin"
    echo "args:" $@
    exit 1
fi

# cwd
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT=$SCRIPT_DIR/fit_from_file.py
echo -ne 'all:\t\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --sec_correction --misid_correction --quiet --fit_systematic 2>err.log

echo -ne 'efficiency:\t'
python $SCRIPT all all $1 --bdt_cut --sec_correction --misid_correction --quiet --fit_systematic 2>err.log

echo -ne 'alt bkg:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --alt_bkg --sec_correction --misid_correction --quiet --fit_systematic 2>err.log

echo -ne 'sec corr:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --misid_correction --quiet --fit_systematic 2>err.log

echo -ne 'misid corr:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --sec_correction --quiet --fit_systematic 2>err.log

echo -ne 'massfit pull:\t'
python $SCRIPT all all $1 --bdt_cut --efficiency --misid_correction --sec_correction --quiet 2>>err.log

