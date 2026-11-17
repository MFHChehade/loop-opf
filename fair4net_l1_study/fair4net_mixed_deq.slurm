#!/bin/bash
# Train one scalar L1-DEQ on an audited mixed-contingency dataset.
#SBATCH -J f4mx_deq
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A YOUR_ALLOCATION

set -euo pipefail
: "${CASE:?Set CASE}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID}"
: "${RUN_KIND:?Set RUN_KIND=smoke or full}"
case "$CASE" in
    case300) BATCH=16 ;;
    case1354pegase|case2869pegase) BATCH=8 ;;
    case9241pegase) BATCH=4 ;;
    *) echo "[mixed-deq][ERROR] unsupported CASE=$CASE" >&2; exit 2 ;;
esac
case "$RUN_KIND" in
    smoke) N=120; EPOCHS=1; STEM="deq_${CASE}_mixed_smoke" ;;
    full) N=10000; EPOCHS=30; STEM="deq_${CASE}_mixed_l1_scalar_e30" ;;
    *) echo "[mixed-deq][ERROR] RUN_KIND must be smoke or full" >&2; exit 2 ;;
esac

ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
OUTROOT="$ROOT/io/fair4net_mixed_contingency_${RUN_KIND}_v1/$CAMPAIGN_ID"
CASE_ROOT="$OUTROOT/$CASE"
DATA="$CASE_ROOT/dataset/dataset_mixed_N${N}_seed0.npz"
ODIR="$CASE_ROOT/deq_l1_scalar_seed0"
CONTRACT="$ODIR/runner_contract.txt"

module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ODIR"
export DEQ_CASE="$CASE" PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

[[ -f "$CASE_ROOT/dataset/.done" && -s "$DATA" ]] || {
    echo "[mixed-deq][ERROR] audited dataset gate is missing" >&2
    exit 3
}
python -u "$STUDY/validate_mixed_multinet.py" \
    --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role dataset --expected-N "$N"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the GH200 job")
print("[cuda]", torch.cuda.get_device_name(0), "torch_cuda=", torch.version.cuda, flush=True)
PY

tmp_contract=$(mktemp "$ODIR/.runner_contract.XXXXXX")
{
    echo "protocol=fair4net_mixed_n1_contingency_v1"
    echo "run_kind=$RUN_KIND"
    echo "case=$CASE"
    echo "dataset_sha256=$(sha256sum "$DATA" | awk '{print $1}')"
    echo "epochs=$EPOCHS"
    echo "batch_size=$BATCH"
    echo "lr=1e-4"
    echo "seed_split=0"
    echo "step_mode=scalar"
    if [[ "$RUN_KIND" == full ]]; then
        echo "picard_iters=50"
        echo "anderson_m=4"
        echo "anderson_lam=1e-3"
        echo "anderson_max_iter=600"
        echo "anderson_tol=1e-10"
    else
        echo "picard_iters=20"
        echo "anderson_m=4"
        echo "anderson_lam=1e-5"
        echo "anderson_max_iter=80"
        echo "anderson_tol=1e-6"
    fi
    echo "w_cost=1"
    echo "w_bal=1e5"
    echo "w_flow=1e8"
    echo "w_flow_l1=1e4"
    echo "trainer_sha256=$(sha256sum "$STUDY/deq_trainer_fair4net.py" | awk '{print $1}')"
} > "$tmp_contract"
if [[ -s "$CONTRACT" ]] && ! cmp -s "$CONTRACT" "$tmp_contract"; then
    echo "[mixed-deq][ERROR] existing runner contract differs; refusing unsafe resume" >&2
    diff -u "$CONTRACT" "$tmp_contract" || true
    rm -f "$tmp_contract"
    exit 4
fi
mv -f "$tmp_contract" "$CONTRACT"

if [[ -f "$ODIR/.done" ]]; then
    python -u "$STUDY/validate_mixed_multinet.py" \
        --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role deq --expected-N "$N"
    echo "[mixed-deq] already complete and validated: $CASE"
    exit 0
fi

printf '%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$(date --iso-8601=seconds)" >> "$ODIR/job_history.tsv"
echo "[mixed-deq] case=$CASE run_kind=$RUN_KIND epochs=$EPOCHS batch=$BATCH job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$RUN_KIND" == smoke ]]; then
    python -u "$STUDY/deq_trainer_fair4net.py" \
        --data "$DATA" --epochs 1 --batch_size "$BATCH" --lr 1e-4 \
        --device cuda --step_mode scalar --seed_split 0 \
        --picard_iters 20 --anderson_m 4 --anderson_lam 1e-5 \
        --anderson_max_iter 80 --anderson_tol 1e-6 --norm_iters 20 \
        --b_anderson_m 4 --b_anderson_lam 1e-4 \
        --b_anderson_max_iter 40 --b_anderson_tol 1e-5 \
        --w_cost 1 --w_bal 1e5 --w_flow 1e8 --w_flow_l1 1e4 --sat_tol_MW 1e-3 \
        --out_dir "$ODIR" --ckpt_name "${STEM}.pt" \
        --require_gs_metadata --no_eval_test_each_epoch --skip_final_test
else
    python -u "$STUDY/deq_trainer_fair4net.py" \
        --data "$DATA" --epochs 30 --batch_size "$BATCH" --lr 1e-4 \
        --device cuda --step_mode scalar --seed_split 0 \
        --picard_iters 50 --anderson_m 4 --anderson_lam 1e-3 \
        --anderson_max_iter 600 --anderson_tol 1e-10 \
        --b_anderson_m 10 --b_anderson_lam 1e-4 \
        --b_anderson_max_iter 100 --b_anderson_tol 1e-6 --norm_iters 100 \
        --w_cost 1 --w_bal 1e5 --w_flow 1e8 --w_flow_l1 1e4 --sat_tol_MW 1e-3 \
        --out_dir "$ODIR" --ckpt_name "${STEM}.pt" \
        --resume --require_gs_metadata --no_eval_test_each_epoch \
        --eval_vs_gt --eval_split test --save_plots
fi

touch "$ODIR/.done"
python -u "$STUDY/validate_mixed_multinet.py" \
    --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role deq --expected-N "$N"
echo "[mixed-deq] PASS case=$CASE end=$(date --iso-8601=seconds)"
