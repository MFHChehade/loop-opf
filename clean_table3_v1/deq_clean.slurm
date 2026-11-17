#!/bin/bash
#SBATCH -J cln_deq
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
set -euo pipefail
: "${CASE:?}"
: "${MODE:?}"
ROOT=/path/to/workdir/DEQ_DC_OPF
cd "$ROOT"
case "$CASE" in
  case300) BATCH=16 ;;
  case1354pegase|case2869pegase) BATCH=8 ;;
  case9241pegase) BATCH=4 ;;
  *) exit 2 ;;
esac
case "$MODE" in
  smoke) EPOCHS=1; PICARD=20; AMAX=80; ATOL=1e-6; ALAM=1e-5; BM=4; BMAX=40; BTOL=1e-5; NORM=20 ;;
  full) EPOCHS=30; PICARD=50; AMAX=600; ATOL=1e-10; ALAM=1e-3; BM=10; BMAX=100; BTOL=1e-6; NORM=100 ;;
  *) exit 2 ;;
esac
SOURCE="$ROOT/io/fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805/$CASE/dataset/dataset_mixed_N10000_seed0.npz"
OUTROOT="$ROOT/io/clean_table3_v1/$CASE"
DATA="$OUTROOT/dataset/dataset_clean_from_mixed_seed0.npz"
OUT="$OUTROOT/deq_${MODE}_seed0"
mkdir -p "$OUT" "$ROOT/logs"
module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
export DEQ_CASE="$CASE" PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
python -u clean_table3_v1/prepare_clean_deq.py --case "$CASE" --source "$SOURCE" --output "$DATA"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'GH200 CUDA unavailable'
print('[cuda]', torch.cuda.get_device_name(0), flush=True)
PY
extra=()
if [[ "$MODE" == smoke ]]; then
  extra+=(--skip_final_test)
else
  extra+=(--eval_vs_gt --eval_split test)
fi
python -u fair4net_l1_study/deq_trainer_fair4net.py \
  --data "$DATA" --epochs "$EPOCHS" --batch_size "$BATCH" --lr 1e-4 \
  --device cuda --step_mode scalar --seed_split 0 \
  --picard_iters "$PICARD" --anderson_m 4 --anderson_lam "$ALAM" \
  --anderson_max_iter "$AMAX" --anderson_tol "$ATOL" \
  --b_anderson_m "$BM" --b_anderson_lam 1e-4 \
  --b_anderson_max_iter "$BMAX" --b_anderson_tol "$BTOL" --norm_iters "$NORM" \
  --w_cost 1 --w_bal 1e5 --w_flow 1e8 --w_flow_l1 1e4 --sat_tol_MW 1e-3 \
  --out_dir "$OUT" --ckpt_name "deq_${CASE}_clean_${MODE}_seed0.pt" \
  --require_gs_metadata --no_eval_test_each_epoch "${extra[@]}"
echo "[clean-deq] PASS case=$CASE mode=$MODE job=$SLURM_JOB_ID"
