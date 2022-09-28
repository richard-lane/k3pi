#!/bin/bash

# Submit all jobs
condor_submit scripts/optimise/condor_files/submit_cf_plus.sub
condor_submit scripts/optimise/condor_files/submit_cf_minus.sub
condor_submit scripts/optimise/condor_files/submit_dcs_plus.sub
condor_submit scripts/optimise/condor_files/submit_dcs_minus.sub
