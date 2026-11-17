#!/bin/bash
# Relative flow violations for Table IV: re-run saved Loop-OPF checkpoints (inference only) and re-score baselines.
#SBATCH -J relviol
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -A YOUR_ALLOCATION
set -euo pipefail
: "${CASE:?Set CASE}"
ROOT=/path/to/workdir/DEQ_DC_OPF
R=$ROOT/rel_violation_v1
module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$R"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=16
python -u eval_relviol.py --case "$CASE" --code "$R/code" --out "$R/results"
