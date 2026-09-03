# LogicMD SCI ViSNet stratified-10 evaluation

This evaluation compares the internally generated LogicMD SCI ViSNet rollout
(`epoch1_step40000`, `inf_step=499`, `ODE=10`) with the paired original LogicMD
MD trajectories using ComplexMD-Eval 0.1.0.

The expanded, topology-corrected ComplexMD-Eval 0.2.0 validation is documented
in `V0.2_VALIDATION.md`; its canonical output directory is `v0.2.0_final/`.

## Fixed sample

The source population is the 50-case LogicMD test panel. Cases are sorted by
PDB ID within each Q1--Q5 reference-motion stratum, then two cases per stratum
are sampled with Python `random.Random(20260903)`. The frozen result is recorded
in `selection_manifest.csv`.

Both reference and generated trajectories contain 500 frames on the documented
1 ns LogicMD/SCI grid. The rewritten XTC header uses a library default and is
not treated as the scientific clock; `dt_ns: 1.0` in the protocols is
authoritative.

## Profiles

- `quick`: first 100 frames; physical checks, paired short-time RMSD, motion,
  RMSF, two-component PCA, and contacts.
- `full`: all 500 frames; the same metric families plus three-component PCA,
  ACF diagnostics through 100 lags, and 10/50/100/250/499 ns windows.

Run from any directory with:

```bash
bash /home/xincheng/xincheng-test/complexmd-eval/evaluations/logicmd_sci_visnet_10_20260903/run.sh
```

Set `COMPLEXMD_EVAL_PYTHON` only when reproducing in another environment that
already contains the package dependencies and MDAnalysis.

## Interpretation boundary

This is a single generated replica against a single reference replica per case.
It measures paired diagnostics and ensemble agreement on the selected panel; it
does not provide publication-grade uncertainty or establish correct transition
rates, MSMs, MFPTs, or Chapman--Kolmogorov consistency.
