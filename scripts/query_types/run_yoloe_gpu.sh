#!/usr/bin/env bash
# All YOLOE work for the WLD experiments, one process at a time on one GPU.
#
#   scripts/query_types/run_yoloe_gpu.sh GPU [REPEAT ...]
#
# 1. Scans with every YOLOE variant. Finished scans are skipped.
# 2. Times the real early stop on five pairs (needs the 'small' scans).
# 3. Runs YOLOE on Experiment 2's narration windows for each REPEAT given
#    (needs that repeat's 'candidates' stage, which Gemini makes).
#
# Every step runs after the previous one, never side by side: processes
# sharing a machine compete for its CPUs, and six scans at once doubled every
# scan's time. GPU is the CUDA device index. Run from the repo root. PYTHON
# picks the interpreter (default: python). Logs go to logs/.
set -euo pipefail

GPU=${1:?usage: run_yoloe_gpu.sh GPU [REPEAT ...]}
shift
PYTHON=${PYTHON:-python}
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs

step() {  # step LOG COMMAND...: run one step, stop everything if it fails
  local log=$1
  shift
  echo "$(date '+%F %T') $*"
  "$@" > "$log" 2>&1 || { echo "Failed; see $log" >&2; exit 1; }
}

for variant in small large small_vocab large_vocab; do
  step "logs/yoloe_scan_$variant.log" "$PYTHON" -m scripts.query_types.wld_yoloe scan --variant "$variant"
done
step logs/yoloe_validate.log "$PYTHON" -m scripts.query_types.wld_yoloe validate
for repeat in "$@"; do
  step "logs/yoloe_windows_run$repeat.log" \
    "$PYTHON" -m scripts.query_types.wld_localization run --repeat "$repeat" --stage yoloe_windows
done
echo "YOLOE work done."
