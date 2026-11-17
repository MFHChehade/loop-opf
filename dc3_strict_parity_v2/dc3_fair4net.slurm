#!/bin/bash
# Official DC3 engine with the documented Fair4Net DC problem adapter.
#SBATCH -J dc3_ours
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
ROOT=/path/to/workdir/DEQ_DC_OPF
cd "$ROOT"
source /path/to/workdir/venv_deq/bin/activate
module load cuda/12.6

SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}
EPOCHS=${EPOCHS:-1000}
PROTOCOL=${PROTOCOL:-mixed}
DATA=${DATA:?set DATA to an audited Fair4Net .npz file}
if [[ -n "${OUT_EXACT:-}" ]]; then
  OUT=$OUT_EXACT
else
  OUT=${OUT_ROOT:-$ROOT/io/dc3_strict_parity_v2/fair4net_${PROTOCOL}_full}/seed_${SEED}
fi
mkdir -p "$OUT" logs

python -u dc3_strict_parity_v2/run_fair4net.py \
  --data "$DATA" --protocol "$PROTOCOL" --seed "$SEED" \
  --epochs "$EPOCHS" --out "$OUT"
