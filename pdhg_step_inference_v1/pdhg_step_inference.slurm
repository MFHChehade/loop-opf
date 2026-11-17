#!/bin/bash
#SBATCH -J pdhg_steps300
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err

set -euo pipefail
ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
MODE=${MODE:-smoke}
CASE=${CASE:?set CASE to case1354pegase, case2869pegase, or case9241pegase}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
DATA=${DATA:-$ROOT/io/fair4net_l1_v1/$CASE/N10000_seed0/random/dataset_random_gsaware.npz}
AUDIT=${AUDIT:-$ROOT/io/fair4net_l1_v2_phaseaware/audits/$CASE/dataset_audit.json}
SOURCE_STEM=${SOURCE_STEM:-$ROOT/io/fair4net_l1_v2_phaseaware/runs/$CASE/deq_l1_scalar_seed0/deq_${CASE}_l1_scalar_v2}
export DEQ_CASE="$CASE"

source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p logs "$OUT_DIR"
test -s "$DATA"
test -s "$AUDIT"
test -s "${SOURCE_STEM}.pt"
test -s "${SOURCE_STEM}_summary.json"
python -c 'import torch; assert torch.cuda.is_available(); print("[cuda]", torch.cuda.get_device_name(0), torch.version.cuda)'

ARGS=(--study-dir "$STUDY" --case-name "$CASE" --data "$DATA" --audit "$AUDIT" --source-stem "$SOURCE_STEM" --out-dir "$OUT_DIR" --device cuda)
if [[ "$MODE" == "smoke" ]]; then
  ARGS+=(--smoke --max-test-samples 64 --depths 1,4,16 --budgets-ms 1,10 --timing-batch-size 16 --quality-batch-size 16 --warmup 1 --repeats 2)
elif [[ "$MODE" == "full" ]]; then
  ARGS+=(--depths "${DEPTHS:-1,2,4,8,16,32,64,128,200}" --budgets-ms "${BUDGETS_MS:-1,2,5,10,20,50,100}" --timing-batch-size "${TIMING_BATCH_SIZE:-16}" --quality-batch-size "${QUALITY_BATCH_SIZE:-16}" --warmup "${WARMUP:-5}" --repeats "${REPEATS:-30}")
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi
python -u pdhg_step_inference_v1/compare_steps.py "${ARGS[@]}"
