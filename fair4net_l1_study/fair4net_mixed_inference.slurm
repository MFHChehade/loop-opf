#!/bin/bash
#SBATCH -J mixed_inf
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%A_%a.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
CAMPAIGN="$ROOT/io/fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805"
OUTROOT="$ROOT/io/fair4net_mixed_contingency_inference_v1"
RUN_MODE=${RUN_MODE:?RUN_MODE must be smoke or full}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}
CASES=(case300 case1354pegase case2869pegase case9241pegase)
CASE=${CASES[${SLURM_ARRAY_TASK_ID:?array task id is required}]}
CASE_ROOT="$CAMPAIGN/$CASE"
DATA="$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.npz"
AUDIT="$CASE_ROOT/dataset/dataset_audit.json"
META="$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.meta.json"
STEM="$CASE_ROOT/deq_l1_scalar_seed0/deq_${CASE}_mixed_l1_scalar_e30"
OUT="$OUTROOT/$RUN_ID/$RUN_MODE/$CASE"

module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p "$OUT" "$ROOT/logs"
export DEQ_CASE="$CASE" PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "[mixed-inference] mode=$RUN_MODE case=$CASE job=${SLURM_JOB_ID:-local} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

python -u "$STUDY/validate_mixed_multinet.py" \
  --root "$CAMPAIGN" --case "$CASE" --mode full --role deq

COMMON=(
  --case "$CASE" --data "$DATA" --audit "$AUDIT" --meta "$META"
  --checkpoint "${STEM}.pt" --summary "${STEM}_summary.json"
  --out-dir "$OUT" --device cuda --warmup 1 --repeats 3
)

if [[ "$RUN_MODE" == smoke ]]; then
  python -u "$STUDY/mixed_deq_inference.py" "${COMMON[@]}" \
    --batch-sizes 1,8 --safe-batch-candidates 8 \
    --max-test-samples 12 --smoke
  python -u "$STUDY/mixed_solver_runtime.py" \
    --case "$CASE" --data "$DATA" --meta "$META" --out-dir "$OUT" \
    --warmup 1 --repeats 3 --samples-per-category 1 --smoke
  python -u "$STUDY/validate_mixed_inference_outputs.py" \
    --case "$CASE" --out-dir "$OUT" --smoke
elif [[ "$RUN_MODE" == full ]]; then
  python -u "$STUDY/mixed_deq_inference.py" "${COMMON[@]}" \
    --batch-sizes 1,8,32 --safe-batch-candidates 64,128,256
  python -u "$STUDY/mixed_solver_runtime.py" \
    --case "$CASE" --data "$DATA" --meta "$META" --out-dir "$OUT" \
    --warmup 1 --repeats 3 --samples-per-category 1
  python -u "$STUDY/validate_mixed_inference_outputs.py" \
    --case "$CASE" --out-dir "$OUT"
else
  echo "invalid RUN_MODE=$RUN_MODE" >&2
  exit 2
fi

scontrol show job "${SLURM_JOB_ID}" > "$OUT/slurm_job_start.txt" 2>&1 || true
(cd "$OUT" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$OUT" && sha256sum -c SHA256SUMS)
echo "MIXED_INFERENCE_COMPLETE mode=$RUN_MODE case=$CASE out=$OUT"
