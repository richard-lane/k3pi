Things to run the entire analysis with
----

Snakemake
----
The analysis chain uses `snakemake` to automate all the steps,
including:
    - creating the dataframes from ROOT files/analysis productions
    -

Prerequisites
----
You'll need the following things to make it work:

### The right python environment
    - hopefully you can just set this up by using `requirements.txt`
    - since the analysis productions live on lxplus, you'll need to set this up there
    - also I might need to figure out a way to either transport the python env to the grid
      for running as a job, or a way to set up python as the first step in the job

### Creating Dataframes
    - AmpGen:
        - Create ROOT files yourself with AmpGen
        - Give them the right name
        - Move them to the root of the git repo, I guess
    - Otherwise:
        - be on lxplus, or somewhere that can see the analysis productions in /eos/

