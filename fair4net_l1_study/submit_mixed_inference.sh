#!/bin/bash
set -euo pipefail

ROOT=/path/to/workdir/DEQ_DC_OPF
cd "$ROOT"
mkdir -p logs io/fair4net_mixed_contingency_inference_v1
source /path/to/workdir/venv_deq/bin/activate

# TACC's sbatch wrapper emits a login/allocation banner on stdout even with
# --parsable. Preserve that diagnostic text on stderr, but return only the last
# numeric SLURM id to callers using command substitution.
submit_job_id() {
  local output job_id
  output=$(sbatch --parsable "$@" 2>&1)
  printf '%s\n' "$output" >&2
  job_id=$(printf '%s\n' "$output" | awk '
    /^[[:space:]]*[0-9]+([;][^[:space:]]+)?[[:space:]]*$/ {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      sub(/;.*/, "")
      id=$0
    }
    END {if (id == "") exit 1; print id}
  ')
  [[ "$job_id" =~ ^[0-9]+$ ]]
  printf '%s\n' "$job_id"
}

ACTION=${1:-smoke}
if [[ "$ACTION" == smoke ]]; then
  SMOKE=$(submit_job_id --array=0-3%2 --export=ALL,RUN_MODE=smoke \
    fair4net_l1_study/fair4net_mixed_inference.slurm)
  printf 'SMOKE_ARRAY=%s\n' "$SMOKE"
  squeue -j "$SMOKE" -o '%.18i %.22j %.2t %.10M %.10l %.40R'
  exit 0
fi

if [[ "$ACTION" != promote || $# -ne 2 ]]; then
  echo "usage: $0 smoke | promote SMOKE_ARRAY_ID" >&2
  exit 2
fi

SMOKE=$2
for CASE in case300 case1354pegase case2869pegase case9241pegase; do
  OUT="$ROOT/io/fair4net_mixed_contingency_inference_v1/$SMOKE/smoke/$CASE"
  python -u fair4net_l1_study/validate_mixed_inference_outputs.py \
    --case "$CASE" --out-dir "$OUT" --smoke
  (cd "$OUT" && sha256sum -c SHA256SUMS)
done
sacct -j "$SMOKE" -X -P --format=JobID,JobName,State,Elapsed,ExitCode,Account,NodeList \
  > "$ROOT/io/fair4net_mixed_contingency_inference_v1/$SMOKE/smoke_slurm_accounting.csv"

FULL=$(submit_job_id --array=0-3%2 \
  --export=ALL,RUN_MODE=full,RUN_ID=$SMOKE \
  fair4net_l1_study/fair4net_mixed_inference.slurm)
REPORT=$(submit_job_id --dependency=afterok:$FULL \
  --export=ALL,RUN_ID=$SMOKE,FULL_JOB=$FULL \
  fair4net_l1_study/finalize_mixed_inference.slurm)

printf 'SMOKE_ARRAY=%s\nFULL_ARRAY=%s\nREPORT_JOB=%s\n' "$SMOKE" "$FULL" "$REPORT"
squeue -j "$FULL,$REPORT" -o '%.18i %.22j %.2t %.10M %.10l %.40R'
