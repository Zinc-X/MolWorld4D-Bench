# Full protocol input contract

Each case declares one topology, one or more independent reference trajectories,
and one or more prediction groups. Atom order and identity must match.

Trajectory strings remain supported. An object entry enables auxiliary data:

```yaml
reference:
  dt_ns: 0.01
  trajectories:
    - path: md_seed0.xtc
      observables: md_seed0_observables.csv
      runtime_s: 86400
      hardware: 1x NVIDIA A100 + 16 CPU cores
      precision: mixed
      batch_size: 1

predictions:
  - name: model_a
    dt_ns: 0.01
    trajectories:
      - path: model_seed0.xtc
        observables: model_seed0_observables.csv
        runtime_s: 120
        hardware: 1x NVIDIA A100
        precision: fp32
        batch_size: 1
```

Observable rows must align one-to-one with trajectory frames before subsampling.
If `time_ns` is present, it must match protocol `dt_ns` and `frame_stride`.

Important full-profile controls include:

```yaml
analysis:
  comparison_horizon: common
  windows_ns: auto
  auto_windows:
    mode: fractions
    fractions: [0.1, 0.25, 0.5, 0.75, 1.0]
  tica_lag_ns: auto
  tica_input_components: 10
  tica_components: 3
  n_states: auto
  n_states_min: 2
  n_states_max: 12
  min_frames_per_state: 20
  min_lagged_samples_per_state: 5
  msm_lags_ns: auto
  acf_max_lag_ns: auto
  min_transition_events: 5
  rare_transition_max_count: 3
  fes_bins: 24
  temperature_K: 300
  structural_stride: 10
  interaction_stride: 2
  solvent_stride: 2
  bootstrap_samples: 500
  bootstrap_confidence: 0.95
  bootstrap_block_ns: auto
  random_seed: 20260903
  trust_threshold_ratio: 2.0
  trust_persistence_frames: 5
```

Physical-time settings accept a positive ns value or `auto`. Automatic TICA,
ACF, and bootstrap controls are duration-relative; automatic MSM lags are
multiples of the resolved TICA lag. Every conversion is frozen in
`parameter_resolution.csv`, including resolved frame counts and rounding error.
Legacy retained-frame settings remain supported for protocol replay.

## Timebases and horizons

`comparison_horizon` controls which portion is evaluated:

- `common`: truncate all trajectories to their shared available duration;
- `reference`: use the primary reference duration and require prediction coverage;
- `prediction`: use the prediction duration and require reference coverage;
- `full_each`: retain each full duration after clock harmonization.

Different uniform `dt_ns` values are exactly downsampled to their smallest
commensurate common grid. Coordinates are never interpolated. The evaluator
rejects non-uniform clocks or a grid exceeding `max_timebase_coarsening`.

## Scoped overrides

An `analysis` mapping may be placed globally, on a case, or on a prediction
group. Mappings are deeply merged in that order, including nested `physical`
and `metrics` settings. The complete case/model result is written to
`analysis_resolution.json`.

For an existing benchmark runtime table, runtime metadata can also be joined by
case without rewriting every trajectory entry:

```yaml
analysis:
  efficiency:
    runtime_table: inference_times.csv
    applies_to: prediction
    case_column: case_id
    runtime_column: inference_wall_seconds
    hardware: 1x NVIDIA A100
    precision: fp32
    batch_size: 1
```
