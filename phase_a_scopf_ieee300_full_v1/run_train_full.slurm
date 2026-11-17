#!/bin/bash
#SBATCH -J scopf300_train
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

: "${DATADIR:?DATADIR was not exported by the submit command}"
: "${BENCHFILE:?BENCHFILE was not exported by the submit command}"
PROJECT=/path/to/workdir/phase_a_scopf_ieee300_full_v1
DATA=${DATADIR}/ieee300_phase_a_K32_N10000_seed0.npz
RESULTDIR=${RESULTDIR:-${PROJECT}/results/training/${SLURM_JOB_ID}}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -f "$DATA"
test -f "${DATA%.npz}.audit.json"
test -f "${DATA%.npz}.provenance.json"
test -f "$BENCHFILE"

module purge
module load gcc/13.2.0 cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
mkdir -p "$RESULTDIR"
cd "$PROJECT"

read -r BATCH_SIZE EVAL_BATCH_SIZE < <(
  python - "$BENCHFILE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("status") != "PASS":
    raise SystemExit("batch benchmark did not pass")
print(result["recommended_batch_size"], result["recommended_eval_batch_size"])
PY
)

RESUME_ARGS=()
if [[ -f "$RESULTDIR/deq_progress.pt" || -f "$RESULTDIR/baseline_progress.pt" ]]; then
  RESUME_ARGS=(--resume)
fi

echo "DATA=$DATA"
echo "RESULTDIR=$RESULTDIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python train_phase_a_full.py \
  --data "$DATA" \
  --out-dir "$RESULTDIR" \
  --mode train \
  --device cuda \
  --epochs 30 \
  --batch-size "$BATCH_SIZE" \
  --baseline-batch-size 1024 \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-every 5 \
  --seed 0 \
  "${RESUME_ARGS[@]}"

python train_phase_a_full.py \
  --data "$DATA" \
  --out-dir "$RESULTDIR" \
  --mode verify \
  --device cuda \
  --eval-batch-size "$EVAL_BATCH_SIZE"

python - "$RESULTDIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = json.loads((root / "production_results.json").read_text())
reload = json.loads((root / "checkpoint_reload.json").read_text())
manifest = json.loads((root / "production_training_manifest.json").read_text())
if result.get("status") != "PRODUCTION_TRAIN_PASS" or not all(result.get("gates", {}).values()):
    raise SystemExit(f"training gates failed: {result.get('gates')}")
if reload.get("status") != "PRODUCTION_CHECKPOINT_RELOAD_PASS":
    raise SystemExit(f"reload failed: {reload.get('status')}")
if manifest.get("status") != "PASS":
    raise SystemExit(f"training manifest failed: {manifest.get('status')}")
print("PRODUCTION_TRAIN_AND_RELOAD_PASS")
PY

(
  cd "$RESULTDIR"
  sha256sum \
    deq_final.pt \
    baseline_final.pt \
    production_results.json \
    checkpoint_reload.json \
    production_training_manifest.json \
    > SHA256SUMS.txt
)

ARCHIVE=${PROJECT}/results/phase_a_ieee300_K32_N10000_train-${SLURM_JOB_ID}.tgz
tar -C "$(dirname "$RESULTDIR")" -czf "$ARCHIVE" "$(basename "$RESULTDIR")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
echo "PERSISTENT_TRAIN_DIR=$RESULTDIR"
echo "TRAIN_ARCHIVE=$ARCHIVE"
