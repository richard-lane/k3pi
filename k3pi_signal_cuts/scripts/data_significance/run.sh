#/bin/bash

# Exit immediately if any script exits non-zero
# Print commands and their arguments as they are executed
set -ex

# Unpack python
mkdir python
tar -xzf python.tar.gz --directory=python
source python/bin/activate

# Clone analysis stuff
git clone https://github.com/richard-lane/k3pi.git
cd k3pi/

# Move the BDT to the right dir
mkdir k3pi_signal_cuts/classifiers/
mv ../2018_dcs_magdown.pkl k3pi_signal_cuts/classifiers/

# Build the shared libraries that we'll need later for phsp bins
./k3pi_efficiency/lib_efficiency/amplitude_models/build.sh

# Create real data dataframes now as well; this is also slow
python k3pi-data/create_real.py 2018 dcs magdown --n_procs 4
pid1=$!
python k3pi-data/create_real.py 2018 cf magdown --n_procs 4
pid2=$!

wait pid1
wait pid2

# Add phsp information to the real data dfs
python k3pi-data/scripts/add_phsp_bins.py 2018 dcs magdown
pid1=$!
python k4pi-data/scripts/add_phsp_bins.py 2018 cf magdown
pid2=$!

wait pid1
wait pid2

# Run the script
python k3pi_signal_cuts/scripts/data_signal_significances.py 2018 magdown

# Don't think this is necessary
cp *.png ..
cd ..

