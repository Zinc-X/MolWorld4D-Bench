# Metric inventory and estimability

`full` enables every implemented module. It does not promise that every metric
is estimable from every input. `metric_coverage.csv` records scalar coverage and
`module_status.csv` records `ok`, `partial`, or `not_estimable` with a reason.

## Core geometry and motion

- finite coordinates, inferred/provided bond and ligand-angle validity, clashes;
- both strict all-bonds/all-angles frame validity and size-robust configurable
  bond/angle-valid-fraction frame validity (`physical_soft_valid_frame_rate`);
- paired backbone/protein/ligand RMSD at common physical times;
- RMSD to frame zero, ligand stay, backbone overmove and drift;
- backbone RMSF correlation/amplitude and per-atom profiles;
- reference-fitted PCA Gaussian W2 and saved projections;
- protein-ligand atom-contact correlation, RMSE, F1, and pair profiles;
- PCA1/contact-count ACF RMSE and saved ACF curves.

## Slow modes, states, kinetics, and free energy

Reference MD defines the representation and prediction MD is projected without
refitting. The module produces:

- reference-fitted Cartesian-CA PCA preprocessing and TICA projections;
- TICA eigenvalues/timescales, TICA Gaussian W2, and VAMP-2 scores;
- deterministic reference-fitted clustering, state occupancy JSD and coverage;
- transition count/probability matrices, Chapman--Kolmogorov RMSE;
- implied-timescale curves at configured lags;
- transition statistics and MFPT have separate estimability gates; MFPT also
  requires connected, sampled state paths in both reference and prediction;
- transition rates and rare-transition type precision/recall;
- two-dimensional TICA free-energy grids, JSD, RMSE, coverage, and out-of-domain
  prediction fraction.

These are finite-sample estimates, not automatic proof of Markovianity. Inspect
reference event counts, CK error, implied-timescale convergence, and reference
replica baselines before making kinetic claims.

## Torsions and structural observables

- Ramachandran phi/psi marginal and joint JSD;
- per-residue phi/psi and chi1 profiles plus chi1 rotamer JSD;
- ligand graph-dihedral distributions for non-ring bridge bonds;
- mass-weighted protein/ligand radius of gyration;
- MDTraj Shrake--Rupley protein/ligand SASA;
- MDTraj DSSP helix/sheet/coil fractions and JSD.

SASA/DSSP require the `advanced` extra and a supported explicit topology.

## Chemical interactions and solvent

- protein-ligand heavy-atom hydrogen-bond proxy;
- geometry-resolved hydrogen bonds when bonded hydrogens are present;
- ligand-element salt proxy and charge-resolved salt bridges when charges exist;
- hydrophobic contacts;
- protein/ligand ring-geometry pi contacts;
- water-bridge heavy-atom proxy and ligand solvent-shell coordination.

Proxy metrics remain named as proxies. Water metrics require explicit solvent;
stripped protein-ligand trajectories correctly return `not_estimable`.

## Thermodynamics and efficiency

Optional per-frame CSV columns use canonical units:

```text
time_ns
potential_energy_kJ_mol
kinetic_energy_kJ_mol
total_energy_kJ_mol
temperature_K
pressure_bar
volume_nm3
```

Shared reference/prediction columns produce mean, Wasserstein, and applicable
per-atom drift comparisons. Coordinates alone cannot determine temperature or
pressure and do not define a force-field potential energy.

Trajectory entries may record `runtime_s`, `hardware`, `precision`, and
`batch_size`. The evaluator reports simulated ns/day and wall-clock speedup only
when the corresponding runtime is supplied. Hardware metadata must accompany
cross-method efficiency claims.

## Replicas, uncertainty, and windows

- every reference-replica pair is written to `reference_baseline.csv`;
- a replica-normalized error curve and persistent trust horizon require at
  least two independent references;
- moving-block bootstrap intervals are emitted per trajectory for time-series
  means;
- case-bootstrap median intervals summarize the benchmark panel;
- cumulative windows include motion/validity plus RMSF, contacts, PCA, FES,
  state occupancy, and state-transition comparison.

One trajectory replayed twice is not two independent replicas.

## Detailed outputs

The main long-form files are `paired_curves.csv`, `rmsf_profiles.csv`,
`contact_profiles.csv`, `pca_projections.csv`, `acf_curves.csv`,
`tica_projections.csv`, `state_occupancy.csv`, `state_transitions.csv`,
`implied_timescales.csv`, `fes.csv`, `torsion_profiles.csv`,
`structural_timeseries.csv`, `interaction_timeseries.csv`,
`solvent_timeseries.csv`, `thermodynamic_timeseries.csv`,
`window_advanced.csv`, `case_uncertainty.csv`, `bootstrap_ci.csv`, and
`efficiency.csv`. Files whose modules are not estimable may be absent; their
reason remains in `module_status.csv`.
