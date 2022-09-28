k3pi
====
Analysis chain for Bristol D0->K3pi analysis

Submodules
----
These aren't proper git submodules, they're just directories
because I thought that would be less confusing

### k3pi-data
Convert analysis production ROOT files to pickle dumps of pandas dataframes, so that they can be easily manipulated with python.

This isn't strictly necessary (we could just deal with the ROOT files directly), but it makes it easier to consistently
apply cuts and makes bugs less likely (e.g. swapping K+3pi and
K-3pi or CF and DCS).

Must be run on lxplus to get the dataframes (since the ROOT files are stored on lxplus);
I've been downloading these dataframes to my laptop for day-to-day development since developing on lxplus can be a pain.

### k3pi_signal_cuts
"BDT cuts" for removing background from the data

### k3pi_efficiency
Efficiency correction - correct for the decay time and phase space
efficiencies

### k3pi_mass_fit
Mass fit - to find out how much signal/background there is

### k3pi_fitter
Fit to decay times to extract mixing parameters, charm interference factor

Python
----
A the scripts in this repo require a few non-standard libraries (`uproot`, `tqdm`);
a full\* list is in [`requirements.txt`](requirements.txt)

\* might not be full yet

### Python requirements
Install requirements with `pip install -r requirements.txt`.

You might want to set up a separate conda environment
(with e.g. `conda create`) for your d->k3pi python installation,
because other projects you work on might require different
packages.

### Running on lxplus
I don't think there are any pre-built environments on `lxplus.cern.ch` that come with all of the requirements already installed,
so you may want to install your own environment from source.
e.g. with miniconda

#### Miniconda install instructions
The install instructions below are for python 3.8 - other versions may work, but this is the one I've been using.

Basically log in to lxplus and follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

You will probably want to do this in your `/work/` partition on lxplus, since your home directory will get full very quickly.

In short:
 - download miniconda
   - `wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh`
 - install it
   - `bash Miniconda3-py38_4.12.0-Linux-x86_64.sh`
 - follow the prompts; you can choose your install location from here as well
 - To activate your miniconda environment:
   - `source $MY_MINICONDA_DIR/bin/activate`
   - where `$MY_MINICONDA_DIR` is the location of your miniconda installation
 - to install requirements:
   - `pip install -r requirements.txt`


