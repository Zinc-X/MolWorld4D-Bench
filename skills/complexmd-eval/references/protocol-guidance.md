# Protocol and interpretation guidance

## Input contract

Each case has one topology, one or more reference trajectories, and one or more
named prediction groups. Every group declares `dt_ns`; all trajectories within
one case must use matching atom order, names, residue names, and residue IDs.

Use NPZ for dependency-light exchange or PDB plus an MDAnalysis-readable
trajectory for standard MD files. See the project `README.md` and
`schemas/protocol.schema.json` for maintained details.

## Profile routing

Choose `screen`, `structural`, `ensemble`, `kinetics`, or `long_horizon` for the
claim being tested; use `full` only when every module is needed. `quick` is a
legacy 100-retained-frame compatibility profile.

Use physical-time controls (`tica_lag_ns`, `msm_lags_ns`, `acf_max_lag_ns`, and
`bootstrap_block_ns`). `auto` is duration-relative and its resolved value must be
reported from `parameter_resolution.csv`. A long trajectory does not justify a
long-timescale kinetic claim without adequate transitions.

## Comparability

Separate result tables when any of these resolved properties change:

- common time grid or comparison horizon;
- resolved physical windows;
- atom representation or topology;
- ligand/protein selection;
- thermodynamic condition;
- conditional trajectory versus unordered equilibrium sampling task.

Uniform commensurate clocks are exactly downsampled without coordinate
interpolation. Non-uniform clocks or excessive coarsening are rejected. Use
`comparison_horizon: common` unless a reference-, prediction-, or full-duration
claim is explicitly intended.

Global, case, and prediction `analysis` mappings are deeply merged in that order.
Use scoped overrides for genuine system/model differences and inspect the frozen
result in `analysis_resolution.json`.

## Results

Start with `failures.csv`, then check `physical_valid_frame_rate`. Low
`overmove_bb` or RMSF amplitude ratio indicates freezing/oversmoothing; high
values indicate overmotion or drift. RMSF correlation describes where motion
occurs and does not establish that its amplitude is correct.

PCA W2 is a reference-fitted ensemble diagnostic, not evidence of correct
kinetics. Contact correlation/F1 describes interaction occupancy but does not
prove a correct binding or escape rate. ACF metrics are meaningful only for an
ordered, time-calibrated prediction.

For publication-grade uncertainty, use multiple independent reference and model
replicas. Inspect `reference_baseline.csv`, `trust_curves.csv`,
`case_uncertainty.csv`, and `bootstrap_ci.csv`; same-run replays are not
independent replicas.

## Input-dependent modules

Per-frame thermodynamic observables must be supplied as canonical CSV columns;
coordinates alone do not determine them. Water metrics require explicit solvent
atoms. Runtime comparisons require `runtime_s` and hardware/precision metadata.
See `references/advanced-guidance.md` before interpreting these modules.
