Things to run the entire analysis with
----

Snakemake
----
The analysis chain uses `snakemake` to automate all the steps,
including:
    - getting the right data from the analysis productions on lxplus/the grid
    - creating the dataframes from ROOT files/analysis productions
    - consistency checks
    - The analysis:
        - train classifier for BDT cut
        - train reweighters for efficiency correction
        - perform mass fits
        - perform time fit

Prerequisites
----
You'll need the following things to make it work:

### The right python environment
    - hopefully you can just set this up by using `requirements.txt`
    - since the analysis productions live on lxplus, you'll need to set this up there
    - also I might need to figure out a way to either transport the python env to the grid
      for running as a job, or a way to set up python as the first step in the job

### Data
    - AmpGen:
        - Create ROOT files yourself with AmpGen
        - Give them the right name
        - Move them to the root of the git repo
    - Otherwise:
        - be on lxplus, or somewhere that can see the analysis productions in /eos/

