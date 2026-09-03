# Protocol and interpretation guidance

## Input contract

Each case has one topology, one or more reference trajectories, and one or more
named prediction groups. Every group declares `dt_ns`; all trajectories within
one case must use matching atom order, names, residue names, and residue IDs.

Use NPZ for dependency-light exchange or PDB plus an MDAnalysis-readable
trajectory for standard MD files. See the project `README.md` and
`schemas/protocol.schema.json` for maintained details.

## Profile routing

`quick` is for checkpoint decisions. Keep cases, starting frames, and model
seeds fixed across checkpoints so changes reflect the model rather than sampling
noise. It intentionally skips ACF and long-window analysis.

`full` is for research conclusions. Set `windows_ns` explicitly, and set
`frame_stride` so the retained temporal resolution remains appropriate for the
fastest process being discussed. A 1-us trajectory does not justify a 1-us
kinetic claim if the reference does not contain enough transitions.

## Comparability

Separate protocols when any of these change:

- generated or reference physical time step;
- total physical window;
- atom representation or topology;
- ligand/protein selection;
- thermodynamic condition;
- conditional trajectory versus unordered equilibrium sampling task.

The current evaluator pairs frames only at common physical times. Distributional
metrics do not require framewise correspondence.

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
