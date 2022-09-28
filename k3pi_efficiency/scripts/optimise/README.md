Reweighter Optimisation
====
Scripts for running the reweighter optimisation, using HTCondor.

To be run on lxplus.


Inputs
----
This optimisation won't "just work" - you'll have to prepare some things first:

 - the python virtualenv
 - the input data
 - `k3pi-data` and `k3pi_efficiency` python libraries


### python virtualenv
We want to bring our python interpreter with us when we run the optimisation
code on the worker nodes using HTCondor.

So far, I have been running python via miniconda.
The entire miniconda installation can be zipped up:
```
tar -zcvf python.tar.gz ../miniconda3/
```
since in my case, the python installation is located in `../miniconda3/`.

Then, in the batch submission script we have instructions to:
 - bring this installation along
   - `transfer_input_files = python.tar.gz` in `submit.sub`.
 - Unzip the python installation
   - `tar -xzf python.tar.gz` in `test.sh` (TODO rename)
 - Activate the python installation, making it the current python interpreter
   - `source miniconda3/bin/activate` in `test.sh` (TODO rename)

### The input data
The reweighter operates on numpy arrays, but to make things easier we'll operate on a pandas DataFrame.
Creation of these is handled by the `k3pi-data` repo [here](https://github.com/richard-lane/k3pi-data).

We want to run on the AmpGen and particle gun data frames for this optimisation - the script `create_dataframes.py`
in this dir handles creating these and putting them in a sensible place.

### libraries
Similarly to the python venv, zip + tar the `k3pi-data/lib_data` and `k3pi_efficiency/lib_efficiency` libraries:
```
tar -zcvf lib_data.tar.gz ../k3pi-data/lib_data
```
```
tar -zcvf lib_efficiency.tar.gz lib_efficiency
```

You'll need to have built the AmpGen amplitude models to do the optimisation - get the source files `cf.cpp`
and `dcs.cpp` from Zenodo (see the README in `lib_efficiency/amplitude_models`), and build using `build.sh`.

## Plotting
Before making plots from the result of the optimisation (`opt_plots.py`), move all the output files from the condor
jobs to a directory called `output/`.
