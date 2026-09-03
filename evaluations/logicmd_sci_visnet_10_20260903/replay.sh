#!/usr/bin/env bash
set -euo pipefail

eval_root="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$eval_root/../.." && pwd)"
eval_python="${COMPLEXMD_EVAL_PYTHON:-/data/xincheng/PVB/.venv/bin/python}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$eval_root/.mplconfig-replay"
mkdir -p "$MPLCONFIGDIR" "$eval_root/replay/quick" "$eval_root/replay/full"

for profile in quick full; do
  "$eval_python" -m complexmd_eval.cli validate \
    "$eval_root/protocol.$profile.yaml" \
    >"$eval_root/replay/$profile/validate.log" 2>&1

  /usr/bin/time -v -o "$eval_root/replay/$profile/time.txt" \
    "$eval_python" -m complexmd_eval.cli run \
      "$eval_root/protocol.$profile.yaml" \
      --out "$eval_root/replay/$profile" \
      >"$eval_root/replay/$profile/stdout.log" \
      2>"$eval_root/replay/$profile/stderr.log"
done

"$eval_python" "$eval_root/verify_replay.py"
