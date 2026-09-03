#!/usr/bin/env bash
set -euo pipefail

eval_root="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$eval_root/../.." && pwd)"
eval_python="${COMPLEXMD_EVAL_PYTHON:-/data/xincheng/PVB/.venv/bin/python}"

export PYTHONPATH="$repo_root/src"
export MPLCONFIGDIR="$eval_root/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$eval_root/quick" "$eval_root/full"

{
  "$eval_python" --version
  "$eval_python" -c 'import complexmd_eval, numpy, pandas, scipy, yaml, matplotlib, MDAnalysis; print("complexmd-eval", complexmd_eval.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("scipy", scipy.__version__); print("PyYAML", yaml.__version__); print("matplotlib", matplotlib.__version__); print("MDAnalysis", MDAnalysis.__version__)'
  uname -a
} > "$eval_root/environment.txt"

sha256sum \
  "$eval_root/protocol.quick.yaml" \
  "$eval_root/protocol.full.yaml" \
  "$eval_root/selection_manifest.csv" \
  /data/xincheng/AtomosVerse/misato_val/logicmd_test50/test50.csv \
  /data/xincheng/PVB/ckpt/uni4d_sci/lig10_visnet/version_0/checkpoint/epoch1_step40000.ckpt \
  /data/xincheng/AtomosVerse/misato_val/logicmd_sci_eval55/pred_sci_visnet_inf499/_cfgs/infer_complex_gpu0.yaml \
  /data/xincheng/AtomosVerse/misato_val/logicmd_sci_eval55/pred_sci_visnet_inf499/process_time_ode10_inf499.csv \
  > "$eval_root/provenance_sha256.txt"

for case_id in 2A14 3L3M 2HMV 2JGS 5FSC 6ABX 3TWP 4K6Z 1M2Q 2UYQ; do
  sha256sum \
    "/data/xincheng/AtomosVerse/misato_val/logicmd_lig10_eval/work/$case_id/${case_id}_sliced.pdb" \
    "/data/xincheng/AtomosVerse/misato_val/logicmd_lig10_eval/work/$case_id/${case_id}_md_sliced.xtc" \
    "/data/xincheng/AtomosVerse/misato_val/logicmd_sci_eval55/pred_sci_visnet_inf499/$case_id/${case_id}_model_ode10_inf499.xtc"
done > "$eval_root/input_sha256.txt"

find "$repo_root/src/complexmd_eval" -type f -name '*.py' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$eval_root/software_sha256.txt"

"$eval_python" -m complexmd_eval.cli validate "$eval_root/protocol.quick.yaml" \
  > "$eval_root/quick/validate.json"
/usr/bin/time -v -o "$eval_root/quick/time.txt" \
  "$eval_python" -m complexmd_eval.cli run "$eval_root/protocol.quick.yaml" --out "$eval_root/quick" \
  > "$eval_root/quick/stdout.log" 2> "$eval_root/quick/stderr.log"

"$eval_python" -m complexmd_eval.cli validate "$eval_root/protocol.full.yaml" \
  > "$eval_root/full/validate.json"
/usr/bin/time -v -o "$eval_root/full/time.txt" \
  "$eval_python" -m complexmd_eval.cli run "$eval_root/protocol.full.yaml" --out "$eval_root/full" \
  > "$eval_root/full/stdout.log" 2> "$eval_root/full/stderr.log"
