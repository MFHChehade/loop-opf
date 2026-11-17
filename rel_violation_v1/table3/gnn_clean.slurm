#!/bin/bash
#SBATCH -J gnn_clean
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -A YOUR_ALLOCATION
set -uo pipefail
module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
ROOT=/path/to/workdir/DEQ_DC_OPF
OUT=$ROOT/rel_violation_v1/table3/gnn_clean
mkdir -p $OUT
cd $ROOT/fair4net_l1_study
export PYTHONHASHSEED=0 OMP_NUM_THREADS=16
for c in case300 case1354pegase case2869pegase case9241pegase; do
  echo "=== $c $(date)"
  python -u $ROOT/rel_violation_v1/table3/gnn_clean.py $c $OUT/$c || echo "[FAIL] $c"
done
echo DONE
