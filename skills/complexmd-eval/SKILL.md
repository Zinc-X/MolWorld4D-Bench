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

## Choose a profile

- Use `quick` while training: a fixed case panel, short trajectories, physical
  checks, motion calibration, RMSF, PCA, and protein-ligand contacts.
- Use `full` for research evaluation from tens of ns to microseconds: retain the
  core metrics and add slow modes/state models, FES, torsions, SASA/DSSP,
  classified interactions, optional thermodynamics, replica baselines,
  uncertainty, efficiency, and cumulative physical-time windows.

Read [references/protocol-guidance.md](references/protocol-guidance.md) when
creating or reviewing a protocol, interpreting missing metrics, or deciding
whether two runs are comparable.

Read [references/advanced-guidance.md](references/advanced-guidance.md) before
making kinetic, thermodynamic, solvent, or efficiency claims.

## Workflow

1. Confirm topology, atom mapping, ligand resname, physical `dt_ns`, and the
   intended profile. Protocol time overrides trajectory-header defaults.
2. Validate before computing:

   ```bash
   complexmd-eval validate protocol.yaml
   ```

3. Run without changing scientific fields after validation:

   ```bash
   complexmd-eval run protocol.yaml --out evaluation/run-name
   ```

4. Inspect `failures.csv`, `module_status.csv`, and `metric_coverage.csv` before
   reading medians.
5. Interpret `report.md` together with `per_case.csv`, `motion_curves.csv`, and
   `window_metrics.csv`. Keep physical validity, motion amplitude, ensemble
   agreement, temporal agreement, and interaction fidelity as separate claims.

## Invariants

- Compare only identical clocks, physical windows, atom representations, and
  paired case sets in one table.
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
