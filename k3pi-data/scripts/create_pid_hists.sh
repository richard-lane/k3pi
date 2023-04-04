#!/bin/bash
# Create pidcalib2 hists, using what I think are the correct cuts

# 2018 magdown
lb-conda pidcalib pidcalib2.make_eff_hists --sample Turbo18 --magnet down --particle Pi --pid-cut "probe_PIDK < 0" --bin-var P --bin-var ETA --max-files 5
lb-conda pidcalib pidcalib2.make_eff_hists --sample Turbo18 --magnet down --particle K --pid-cut "probe_PIDK > 8" --bin-var P --bin-var ETA  --max-files 5
