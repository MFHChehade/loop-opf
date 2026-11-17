#!/bin/bash
#SBATCH -J pow2clean
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
BASE="$ROOT/io/fair4net_l1_v2_phaseaware"
DATA="$ROOT/io/fair4net_l1_v1/$CASE/N10000_seed0/random/dataset_random_gsaware.npz"
AUDIT="$BASE/audits/$CASE/dataset_audit.json"
STEM="$BASE/runs/$CASE/deq_l1_scalar_seed0/deq_${CASE}_l1_scalar_v2"
EXTRA=()
if [[ "$CASE" == case300 ]]; then
  STEM="$ROOT/io/case300_gsfix_penalty_ab_v2/new_linear/deq_case300_new_linear_e30_p200_m10"
fi
if [[ "$CASE" == case9241pegase ]]; then
  AUDIT="$ROOT/io/case9241_clean_recovery_v5/dataset_audit_current.json"
  STEM="$ROOT/io/case9241_clean_current_v5/deq_l1_scalar_seed0/deq_case9241pegase_clean_current_v2"
  EXTRA+=(--sidecar "$ROOT/io/case9241_clean_recovery_v5/flow_offset_audited.npy")
  if [[ "$MODE" != smoke ]]; then
    EXTRA+=(--reference-initial "$ROOT/io/case9241_initial_steps_fixed_k_v1/full/quality_by_k.csv")
    EXTRA+=(--reference-learned "$ROOT/io/case9241_strict_fixed_k_v8/full/quality_by_k.csv")
  fi
fi
for path in "$DATA" "$AUDIT" "${STEM}.pt" "${STEM}_summary.json"; do test -s "$path"; done
mkdir -p "$ROOT/logs"
ARGS=(--case "$CASE" --study-dir "$ROOT/fair4net_l1_study" --compare-dir "$ROOT/pdhg_step_inference_v1"
      --data "$DATA" --audit "$AUDIT" --source-stem "$STEM" --batch-size 8)
if [[ "$MODE" == smoke ]]; then
  ARGS+=(--out-root "$ROOT/io/fixed_k_pow2_v1/smoke/studyA" --k-values 1,8 --max-test-samples 12)
elif [[ "$MODE" == parity ]]; then
  ARGS+=(--out-root "$ROOT/io/fixed_k_pow2_v1/parity/studyA" --k-values 1,2,4,8,16,32,64,128,256)
elif [[ "$MODE" == full ]]; then
  ARGS+=(--out-root "$ROOT/io/fixed_k_pow2_v1/studyA" --k-values 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192)
elif [[ "$MODE" == extended ]]; then
  ARGS+=(--out-root "$ROOT/io/fixed_k_pow2_v1/extended/studyA" --k-values 8192,16384)
else
  echo "invalid MODE=$MODE" >&2; exit 2
fi
python -u "$ROOT/fixed_k_pow2_v1/study_a_pow2.py" "${ARGS[@]}" "${EXTRA[@]}"
