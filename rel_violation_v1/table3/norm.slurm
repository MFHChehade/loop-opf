#!/bin/bash
# Normalized metrics for Table III (re-run saved checkpoints, per-line overload / rating) + RMSE scale per network.
#SBATCH -J norm_t3
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -A YOUR_ALLOCATION
set -uo pipefail
R=/path/to/workdir/DEQ_DC_OPF/rel_violation_v1/table3
module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd /path/to/workdir/DEQ_DC_OPF
export PYTHONHASHSEED=0 OMP_NUM_THREADS=16
for C in ${CASES:-case300 case1354pegase case2869pegase case9241pegase}; do
  echo "===== $C $(date)"
  DEQ_CASE=$C python -u "$R/norm_eval.py" --case "$C" --out "$R/results" || echo "[norm] $C FAILED"
done
