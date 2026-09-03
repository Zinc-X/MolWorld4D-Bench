---
name: complexmd-eval
description: >-
  Evaluate generated protein-small-molecule MD trajectories against reference
  MD with the ComplexMD-Eval package. Use for training checkpoint screening,
  research-scale trajectory comparison, physical-time window analysis, or a
  standardized protein-ligand dynamics report including slow modes, state
  kinetics, structural observables, interactions, uncertainty, and efficiency.
  Do not use kinetic modules on unordered equilibrium samples.
---

# ComplexMD-Eval

Use the installed `complexmd-eval` CLI; do not reimplement metrics in the skill.
This skill requires `complexmd-eval>=0.3.1,<0.4.0`.

## Choose a profile

- Use `screen` for low-cost physical and motion checks while training.
- Use `structural` for torsions, Rg/SASA/DSSP, contacts, interactions, and solvent.
- Use `ensemble` for PCA/TICA, state occupancy, and FES.
- Use `kinetics` for ordered-trajectory ACF, VAMP/MSM/ITS/MFPT, and rare events.
- Use `long_horizon` for duration-relative cumulative-window recovery.
- Use `full` when every module is needed. Use `quick` only to replay legacy
  100-retained-frame protocols.

Read [references/protocol-guidance.md](references/protocol-guidance.md) when
creating or reviewing a protocol, interpreting missing metrics, or deciding
whether two runs are comparable.

Read [references/advanced-guidance.md](references/advanced-guidance.md) before
making kinetic, thermodynamic, solvent, or efficiency claims.

## Workflow

1. Check the engine and profile-specific runtime before evaluating:

   ```bash
   complexmd-eval doctor --profile PROFILE
   ```

   If the command is unavailable, explain that the engine must be installed and
   ask before running `python -m pip install "complexmd-eval[all]>=0.3.1,<0.4"`.
   Treat doctor errors as blockers. Treat warnings as explicitly missing optional
   capabilities; do not silently claim the affected metrics.
2. Confirm topology, atom mapping, ligand resname, physical `dt_ns`, intended
   profile, and `comparison_horizon`. Protocol time overrides trajectory headers.
3. Validate before computing:

   ```bash
   complexmd-eval validate protocol.yaml
   ```

4. Run without changing scientific fields after validation:

   ```bash
   complexmd-eval run protocol.yaml --out evaluation/run-name
   ```

5. Inspect `failures.csv`, `module_status.csv`, `timebase_audit.csv`,
   `parameter_resolution.csv`, and `metric_coverage.csv` before reading medians.
6. Interpret `report.md` together with `per_case.csv`, `motion_curves.csv`, and
   `window_metrics.csv`. Keep physical validity, motion amplitude, ensemble
   agreement, temporal agreement, and interaction fidelity as separate claims.

## Invariants

- Permit different clocks only when `timebase_audit.csv` confirms exact common-grid
  downsampling. Never reinterpret an unsafe resampling failure as a missing metric.
- Compare only identical resolved horizons, physical windows, atom
  representations, and paired case sets in one table.
- Protein-align before evaluating ligand pose; never ligand-align away pose drift.
- Paired RMSD is a short-horizon diagnostic, not a long-horizon headline.
- Stability without motion is not success; always inspect `overmove_bb` and
  `rmsf_backbone_amplitude_ratio`.
- Missing or non-estimable metrics stay missing. Never replace them with zero.
- Fit learned coordinates, clusters, bins, and states on reference MD, then
  apply them unchanged to predictions.
- Treat `kinetics_estimable` as a minimum data gate, not proof of Markovianity;
  inspect event counts, CK error, implied-timescale convergence, and reference
  replica baselines.
- Never infer energy, temperature, pressure, hydration, or runtime from missing
  source data. Report the module-status reason.
