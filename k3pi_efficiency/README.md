# k3pi_efficiency
Efficiency Reweighting for LHCb D0 -> K3pi analysis

## Conventions
 - D -> K+ pi- pi- pi+ is assumed; if you want to pass in a conjugate decay (e.g. RS D0->K-3pi or WS Dbar0->K-3pi),
   flip the 3 momenta yourself. Maybe I'll do that for you actually TODO, but then you'll have to pass in D IDs
 - Everything in MeV
 - All decay times in units of D lifetime

## Notes
Need to properly write this out later TODO

 - flipped momenta such that RS is Dbar0 -> K+3pi and WS is D0 -> K+3pi; will need to do something to make this
   consistent with the data as well
 - Momentum ordered based on M(Kpi) for the two equivalent pions
 - Reweight AG->MC times to avoid large weights, then keep those weights for phsp weighting, then final weights
   are phsp weights * 1 / time weights
