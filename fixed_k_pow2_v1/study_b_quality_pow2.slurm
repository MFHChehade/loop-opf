#!/bin/bash
#SBATCH -J pow2mixed
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:10:00
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%A_%a.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/path/to/workdir/DEQ_DC_OPF
cd "$ROOT"
module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
CASES=(case300 case1354pegase case2869pegase case9241pegase)
CASE=${CASES[${SLURM_ARRAY_TASK_ID:?array task id required}]}
if [[ "$CASE" == case1354pegase || "$CASE" == case2869pegase ]]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
export DEQ_CASE="$CASE"
MODE=${MODE:-smoke}
CAMPAIGN="$ROOT/io/fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805"
CASE_ROOT="$CAMPAIGN/$CASE"
STEM="$CASE_ROOT/deq_l1_scalar_seed0/deq_${CASE}_mixed_l1_scalar_e30"
for path in "$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.npz" "$CASE_ROOT/dataset/dataset_audit.json" \
  "$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.meta.json" "${STEM}.pt" "${STEM}_summary.json"; do test -s "$path"; done
mkdir -p "$ROOT/logs"
ARGS=(--case "$CASE" --data "$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.npz"
      --audit "$CASE_ROOT/dataset/dataset_audit.json"
      --meta "$CASE_ROOT/dataset/dataset_mixed_N10000_seed0.meta.json"
      --checkpoint "${STEM}.pt" --summary "${STEM}_summary.json"
      --device cuda --warmup 1 --repeats 3 --batch-sizes 8)
if [[ "$MODE" == smoke ]]; then
  OUT="$ROOT/io/fixed_k_pow2_v1/smoke/studyB/$CASE"
  ARGS+=(--k-values 1,8 --max-test-samples 12 --smoke)
elif [[ "$MODE" == repair_smoke_v2 ]]; then
  OUT="$ROOT/io/fixed_k_pow2_v1/repair_smoke_v2/studyB/$CASE"
  ARGS+=(--k-values 1,8 --max-test-samples 12 --smoke)
  ARGS+=(--reference-quality "$ROOT/io/fixed_k_pow2_v1/smoke/studyB/$CASE/quality_by_k.csv")
elif [[ "$MODE" == full ]]; then
  OUT="$ROOT/io/fixed_k_pow2_v1/studyB/$CASE"
  ARGS+=(--k-values 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192)
  ARGS+=(--reference-quality "$ROOT/io/fair4net_mixed_fixed_k_v1/901788/full/$CASE/quality_by_k.csv")
elif [[ "$MODE" == extended ]]; then
  OUT="$ROOT/io/fixed_k_pow2_v1/extended/studyB/$CASE"
  ARGS+=(--k-values 8192,16384)
  ARGS+=(--reference-quality "$ROOT/io/fair4net_mixed_fixed_k_v1/901788/full/$CASE/quality_by_k.csv")
else
  echo "invalid MODE=$MODE" >&2; exit 2
fi
python -u "$ROOT/fixed_k_pow2_v1/study_b_quality_pow2.py" "${ARGS[@]}" --out-dir "$OUT"
