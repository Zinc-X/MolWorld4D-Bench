# Advanced evaluation guidance

## Start with estimability

Read `module_status.csv` and `metric_coverage.csv`. `not_estimable` is a valid
scientific outcome when required inputs or events are absent. Do not replace it
with a proxy or zero.

## Slow modes and state kinetics

TICA/PCA preprocessing, state centers, and FES bins are reference-fitted. Use
TICA W2 and state occupancy for ensemble claims. Use transition matrices, ITS,
MFPT, rates, and rare-event recovery only for ordered trajectories with a
correct physical clock.

Before a kinetic claim, require adequate reference transition counts, acceptable
reference CK error, approximate implied-timescale convergence across lag choices,
and preferably multiple independent reference replicas. `kinetics_estimable=1`
is only the package's minimum event gate.

## Chemistry and thermodynamics

Names containing `proxy` are deliberately weaker observables. Exact hydrogen
bond geometry requires bonded hydrogens; charge-resolved salt bridges require
charges. SASA/DSSP require MDTraj and a compatible topology. Water metrics
require retained solvent atoms.

Energy, temperature, pressure, and volume are read from aligned per-frame CSV
data. Potential and total energy drift are different physical claims. Do not
describe potential-energy drift as NVE energy conservation.

## Efficiency

Report simulated ns/day with runtime, hardware, precision, batch size, and
trajectory length. A speedup is emitted only when both prediction and reference
runtime are supplied. Cross-hardware speedups should be labelled wall-clock
comparisons, not algorithm-only acceleration.
