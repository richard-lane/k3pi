Amplitude Models
====
If we want to use the DCS and CF models from python, we will need to use a thin wrapper
as its easier to interact with C-compatible types from Python (rather than C++).

Source Code
----
The amplitude source code can be found on Zenodo: `https://zenodo.org/record/3457086`, DOI `10.5281/zenodo.3457086`.

## Building
To build the libraries, copy the source code (`cf.cpp` and `dcs.cpp`) to this dir (`lib_efficiency/amplitude_models/`).

To build the wrapper shared libraries use `build.sh`.
It isn't a complicated script, so if something doesn't work have a look at it and try to find the right command

