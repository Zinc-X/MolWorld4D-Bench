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
  windows_ns: [10, 50, 100, 500, 1000]
  tica_lag_frames: 10
  tica_input_components: 10
  tica_components: 3
  n_states: 6
  msm_lag_frames: [10, 20, 50]
  min_transition_events: 5
  rare_transition_max_count: 3
  fes_bins: 24
  temperature_K: 300
  structural_stride: 10
  interaction_stride: 2
  solvent_stride: 2
  bootstrap_samples: 500
  bootstrap_confidence: 0.95
  bootstrap_block_frames: 20
  random_seed: 20260903
  trust_threshold_ratio: 2.0
  trust_persistence_frames: 5
```

Lag parameters are retained-frame counts. Their physical lag is the parameter
times `dt_ns * frame_stride`.

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
