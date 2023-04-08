Amplitude Models
====
If we want to use the DCS and CF models from python, we will need to use a thin wrapper
as its easier to interact with C-compatible types from Python (rather than C++).

Source Code
----
The amplitude source code can be found on Zenodo: `https://zenodo.org/record/3457086`, DOI `10.5281/zenodo.3457086`,
and is included in this repository so i can make the analysis work on condor

It is from the paper by Tim Evans et al. (can be found using the above DOI) and is licenced under creative commons 4.0

The work here is a python wrapper around a C wrapper than enables the amplitude models to be called from python

## Building
To build the wrapper shared libraries use `build.sh`.
It isn't a complicated script, so if something doesn't work have a look at it and try to find the right command

