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
python k3pi-data/create_ampgen.py ./ws_D02piKpipi.root dcs
python k3pi-data/create_ampgen.py ./rs_Dbar02piKpipi.root cf

# Only create a few pgun dfs for speed
python k3pi-data/create_pgun.py 2018 dcs magdown -n 8 -v
python k3pi-data/create_pgun.py 2018 cf magdown -n 8 -v

python k3pi-data/create_mc.py 2018 dcs magdown
python k3pi-data/create_mc.py 2018 cf magdown

# Only create a few real dfs for speed
python k3pi-data/create_real.py -n 12 2018 cf magdown --n_procs 6
python k3pi-data/create_real.py -n 12 2018 dcs magdown --n_procs 6

