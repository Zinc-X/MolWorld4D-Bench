# MolWorld4D-Bench

MolWorld4D-Bench is a protocol-driven benchmark for generated
protein-small-molecule MD trajectories. Its installable Python package and CLI
are named `complexmd-eval`.

ComplexMD-Eval compares predicted trajectories with reference MD while keeping
physical validity, motion amplitude, ensemble agreement, temporal agreement,
and interaction fidelity as separate scientific claims.

Task-oriented profiles use the same metric implementations:

- `screen`: low-cost physical and motion checks;
- `structural`: torsions, structural observables, interactions, and solvent;
- `ensemble`: PCA/TICA, state occupancy, and FES;
- `kinetics`: ACF, VAMP/MSM/ITS/MFPT, rare events, and replica baselines;
- `long_horizon`: automatic physical-time windows across structural and kinetic metrics;
- `full`: every module; `quick` remains as a 100-frame compatibility profile.

Version 0.3 adds physical-time lag/block controls, exact common-grid resampling,
explicit comparison-horizon policies, automatic duration-relative windows,
case/model analysis overrides, and evidence-limited adaptive state counts. Metrics
that cannot be supported by a particular input are explicitly marked
`not_estimable`.

## Install

From PyPI after the release is published:

```bash
python -m pip install complexmd-eval

# Full structural/interaction stack (MDAnalysis, MDTraj, NetworkX)
python -m pip install "complexmd-eval[all]"
```

From a source checkout for development:

```bash
python -m pip install -e .

# Required only for PDB + XTC/DCD/TRR input
python -m pip install -e '.[trajectory]'

# Full structural/interaction stack (MDAnalysis, MDTraj, NetworkX)
python -m pip install -e '.[all]'
```

Python 3.10 or newer is required.

## Codex Skill and Plugin

The repository is also a Skill-only Codex Plugin. GitHub releases include a
`molworld4d-bench-plugin-<version>.zip` archive. The Skill chooses an
appropriate evaluation profile and enforces the package's scientific
comparability and estimability rules; all numerical work remains in the tested
Python engine.

Check the installed engine and optional capabilities before a run:

```bash
complexmd-eval doctor --profile full
complexmd-eval doctor --protocol protocol.yaml
```

For personal use outside this repository, preview and then install the bundled
Skill. Installation never overwrites an existing different Skill:

```bash
complexmd-eval skill install --scope user --dry-run
complexmd-eval skill install --scope user
complexmd-eval skill status
```

For a repository-scoped installation, use `--scope repo --repo /path/to/repo`.
Use `--mode symlink` only for a development checkout that will remain at a stable
path. Installing the Plugin or Skill does not install scientific Python
dependencies; install `complexmd-eval` or `complexmd-eval[all]` separately.

## Author

[Zinc-X](https://github.com/Zinc-X) ·
[ORCID 0009-0008-6552-3247](https://orcid.org/0009-0008-6552-3247)

## Input formats

### Standard MD formats

XTC, DCD, TRR, and other MDAnalysis-readable trajectories require a topology
file. Coordinates are interpreted in Angstrom. Atom order, atom names, residue
names, and residue IDs must match between reference and prediction.

### Portable NPZ exchange format

NPZ inputs make tests and sharing independent of a particular trajectory stack.
They contain:

```text
positions   float array, shape (frames, atoms, 3), Angstrom
atom_names  string array, shape (atoms,)
resnames    string array, shape (atoms,)
resids      integer array, shape (atoms,)
elements    string array, shape (atoms,)
bonds       integer array, shape (bonds, 2), optional
```

The protocol's `dt_ns` is authoritative. XTC header time is deliberately not
used as the physical clock.

## Run the included demo

```bash
python examples/generate_demo_data.py
complexmd-eval validate examples/protocol.quick.yaml
complexmd-eval run examples/protocol.quick.yaml --out demo-out/quick
complexmd-eval run examples/protocol.kinetics.yaml --out demo-out/kinetics
complexmd-eval run examples/protocol.full.yaml --out demo-out/full
```

To inspect a template before data exist:

```bash
complexmd-eval validate examples/protocol.full.yaml --no-check-paths
```

## Protocol essentials

```yaml
protocol_version: "1.0"
suite: my-complex-suite
profile: full
strict_case_pairing: true

selections:
  ligand_resname: MOL
  alignment: protein_backbone

analysis:
  comparison_horizon: common   # common | reference | prediction | full_each
  frame_stride: 1
  max_frames: null
  windows_ns: auto
  auto_windows:
    mode: fractions
    fractions: [0.1, 0.25, 0.5, 0.75, 1.0]
  ligand_stay_cutoff_A: 5.0
  contact_cutoff_A: 4.5
  contact_frequency_threshold: 0.10
  pca_components: 3
  acf_max_lag_ns: auto
  physical:
    bond_relative_error_max: 0.20
    angle_error_max_rad: 0.35
    clash_distance_A: 1.0
    bond_valid_fraction_min: 0.99
    ligand_bond_valid_fraction_min: 0.90
    angle_valid_fraction_min: 0.90

  tica_lag_ns: auto
  tica_input_components: 10
  tica_components: 3
  n_states: auto
  n_states_min: 2
  n_states_max: 12
  min_frames_per_state: 20
  min_lagged_samples_per_state: 5
  msm_lags_ns: auto
  fes_bins: 24
  structural_stride: 10
  interaction_stride: 2
  bootstrap_samples: 500
  bootstrap_block_ns: auto

cases:
  - id: 1ABC
    topology: /data/1ABC/topology.pdb
    reference:
      dt_ns: 0.01
      trajectories:
        - /data/1ABC/md_replica_0.xtc
        - /data/1ABC/md_replica_1.xtc
    predictions:
      - name: model_a
        dt_ns: 0.01
        analysis:                 # optional model-specific override
          tica_lag_ns: 2.0
          n_states_max: 8
        trajectories:
          - path: /data/1ABC/model_a_seed_0.xtc
            observables: /data/1ABC/model_a_seed_0_observables.csv  # optional
            runtime_s: 120                                         # optional
            hardware: 1x NVIDIA A100                               # optional
          - /data/1ABC/model_a_seed_1.xtc
```

Relative paths are resolved from the protocol file. `frame_stride` changes the
loaded time spacing, while `max_frames` is applied after striding. For very long
trajectories, choose a stride that retains the fastest process being evaluated;
do not use aggressive subsampling for high-frequency ACF claims.

`analysis` may also appear on a case or prediction group. Settings are deeply
merged in `global -> case -> prediction` order. Physical-time parameters are
converted separately on each harmonized case/model clock. Legacy
`*_frames` controls remain supported, but new protocols should use `*_ns`.

## Outputs

```text
run/
├── protocol.resolved.yaml
├── analysis_resolution.json
├── run_manifest.json
├── per_replica.csv
├── metrics_long.csv
├── per_case.csv
├── summary_by_model.csv
├── failures.csv
├── metric_coverage.csv
├── module_status.csv
├── timebase_audit.csv
├── parameter_resolution.csv
├── frame_metrics.csv
├── motion_curves.csv
├── window_metrics.csv       # full profile when windows are configured
├── window_advanced.csv
├── paired_curves.csv
├── rmsf_profiles.csv
├── contact_profiles.csv
├── pca_projections.csv
├── acf_curves.csv
├── tica_projections.csv
├── state_occupancy.csv
├── state_transitions.csv
├── implied_timescales.csv
├── fes.csv
├── torsion_profiles.csv
├── structural_timeseries.csv
├── interaction_timeseries.csv
├── case_uncertainty.csv
├── bootstrap_ci.csv
├── efficiency.csv
├── figures/
└── report.md
```

Failures remain visible in `failures.csv`; missing metrics are not converted to
zero. Always read `metric_coverage.csv` and `module_status.csv` before summaries.
`per_case.csv` aggregates prediction replicas by the median, and
`summary_by_model.csv` aggregates paired cases by the median. Use the long tables
for statistical analysis rather than treating the summary as a single score.

By default, every case must contain the same named prediction models
(`strict_case_pairing: true`). This prevents model summaries from silently using
different case sets.

## Implemented metric families

- finite-frame, bond-deviation, ligand-angle, and steric-clash checks;
- short-horizon paired backbone/protein/ligand RMSD at common physical times;
- backbone and ligand RMSD to reference frame zero, ligand stay, and overmove;
- backbone RMSF correlation and amplitude ratio;
- reference-fitted backbone PCA and Gaussian W2 distance;
- protein-ligand atomic contact frequency correlation, RMSE, and F1;
- PCA-1 and contact-count autocorrelation errors in the `full` profile;
- cumulative physical-time window diagnostics;
- reference-fitted TICA, VAMP-2, state occupancy, MSM transition matrices,
  implied timescales, conditional MFPT/rare-transition estimates, and CK error;
- TICA free-energy surfaces with out-of-reference-domain accounting;
- Ramachandran, chi1 rotamers, ligand graph dihedrals, Rg, SASA, and DSSP;
- hydrogen-bond/salt proxies plus charge/hydrogen-resolved variants when the
  topology supports them, hydrophobic contacts, pi contacts, and water shells;
- optional energy/temperature/pressure time-series comparison;
- reference-replica baselines, block/case bootstrap intervals, replica-normalized
  trust horizons, advanced physical windows, and efficiency metadata.

See [the metric inventory](docs/metrics.md) and [protocol contract](docs/protocol.md)
for exact prerequisites and outputs.

Current physical thresholds are configurable protocol defaults, not universal
chemical constants. For publication, preregister them and validate them on
reference MD replicas.

## Full versus quick

`quick` defaults to the first 100 loaded frames and skips ACF and long-window
analysis. It is intended for detecting training failures: invalid geometry,
freezing, overmotion, loss of flexibility, ensemble drift, and lost contacts.

`full` does not impose a frame cap and enables the advanced analysis stack in
addition to ACF and configured physical-time windows. Long trajectories may
require explicit structural/interaction strides and sufficient memory.

The CLI can override the profile without changing metric definitions:

```bash
complexmd-eval run protocol.yaml --profile quick --out train-eval/step-40000
complexmd-eval run protocol.yaml --profile full --out research-eval/step-40000
```

## Scientific boundaries

- Paired RMSD is a short-horizon diagnostic only.
- Equilibrium-distribution agreement does not establish kinetic correctness.
- A stable but motionless trajectory is detected by overmove and RMSF-amplitude
  ratios; stability alone is not success.
- Commensurate physical clocks are exactly downsampled to a common grid without
  coordinate interpolation. Unsafe or excessive coarsening is rejected.
- Compare only results with the same resolved horizon and windows; inspect
  `timebase_audit.csv` and `parameter_resolution.csv`.
- Sparse-event rates and MFPT are marked non-estimable when transition evidence
  is below the protocol minimum; finite estimates still require CK/ITS review.
- Enhanced-sampling references require correct reweighting and cannot be used as
  unbiased kinetic references by default.
- Coordinates alone cannot determine temperature/pressure or a force-field
  potential energy. Supply canonical observable time series.
- Stripped-solvent inputs cannot support water-bridge or hydration claims.
- Heavy-atom hydrogen-bond and ligand-charge classifications are explicitly
  labelled proxies unless hydrogens/formal charges are supplied.

## Development

```bash
python -m pip install -e '.[test]'
pytest
```

Metric modules live in `metrics.py`, `advanced.py`, `structural.py`,
`interactions.py`, and `observables.py`. New analyses should remain explicit
modules with declared prerequisites and estimability states.
