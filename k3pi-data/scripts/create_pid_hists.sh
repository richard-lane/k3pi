#!/bin/bash
# Create pidcalib2 hists, using what I think are the correct cuts
# This wiki page is probably relevant
# https://twiki.cern.ch/twiki/bin/view/LHCb/PIDCalibPackage

# 2018 magdown
lb-conda pidcalib pidcalib2.make_eff_hists --sample Turbo18 --magnet down --particle Pi --pid-cut "DLLK < 0.0" --bin-var P --bin-var ETA --max-files 5
lb-conda pidcalib pidcalib2.make_eff_hists --sample Turbo18 --magnet down --particle K --pid-cut "DLLK > 8.0" --bin-var P --bin-var ETA  --max-files 5
