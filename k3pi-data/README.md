# k3pi-data
Data parsing, cleaning, etc for D->K3pi analysis to get everything into a consistent format

I've chosen to save everything as pickle dumps of pandas dataframes


## Types of data in this analysis
To create the MC, particle gun or real data dataframes, just run the right script on lxplus.
To create the AmpGen dataframes, create the right D->K3pi ROOT files using AmpGen, and point the script at them.

AmpGen: https://github.com/GooFit/AmpGen  
D->K3pi config files: https://zenodo.org/record/3457086  

### AmpGen
Amplitude models for $D\rightarrow K+3\pi$.  
The only difference between an amplitude and its conjugate
(e.g. $D^0\rightarrow K^+3\pi$ and $\bar{D}^0\rightarrow K^-3\pi$) should be the direction of the 3-momenta,
so if we want to transform between them we just need to multiply the 3-momenta by -1.  
We can do this by using the `flip_momenta` function [here](lib_data/util.py#L31).

**Used for:** training the efficiency reweighter

### MC
Simulation using the full LHCb software.
Generates events according to AmpGen amplitude models

**Used for:** training the BDT cuts

### Particle Gun
Simulation: higher statistics than LHCb MC, but potentially less accurate.
Uses the AmpGen models

**Used for:** training the efficiency reweighter

### Real Data
**Used for:** the analysis

## Creating the dataframes
Run the `create_data.py`, `create_mc.py` scripts etc. to create the dataframes.
I haven't set everything up yet so some options aren't yet available - so far I've mostly been working with
2018 magdown data/MC.

The scripts should output an error message if you use them wrong telling you the allowed options.
It takes a __long__ time to parse/dump some of these dataframes - especially the real data - so it's often best to just
cancel the process before it finishes if you just want to test something/run some scripts/do anything except the full
analysis.
The scripts will pick up where they left off if you do this, so won't waste time creating/dumping dataframes that
already exist.

