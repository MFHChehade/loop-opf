#!/bin/bash
# Run all seven baselines on an audited mixed-contingency dataset.
#SBATCH -J f4mx_base
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION

set -euo pipefail
: "${CASE:?Set CASE}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID}"
: "${RUN_KIND:?Set RUN_KIND=smoke or full}"
case "$CASE" in
    case300|case1354pegase|case2869pegase|case9241pegase) ;;
    *) echo "[mixed-baselines][ERROR] unsupported CASE=$CASE" >&2; exit 2 ;;
esac
case "$RUN_KIND" in
    smoke) N=120 ;;
    full) N=10000 ;;
    *) echo "[mixed-baselines][ERROR] RUN_KIND must be smoke or full" >&2; exit 2 ;;
esac

ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
OUTROOT="$ROOT/io/fair4net_mixed_contingency_${RUN_KIND}_v1/$CAMPAIGN_ID"
CASE_ROOT="$OUTROOT/$CASE"
DATA="$CASE_ROOT/dataset/dataset_mixed_N${N}_seed0.npz"
ODIR="$CASE_ROOT/baselines_seed0"
CONTRACT="$ODIR/runner_contract.txt"

module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ODIR"
export DEQ_CASE="$CASE" PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

[[ -f "$CASE_ROOT/dataset/.done" && -s "$DATA" ]] || {
    echo "[mixed-baselines][ERROR] audited dataset gate is missing" >&2
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
    echo "seed=0"
    echo "epochs=$([[ "$RUN_KIND" == smoke ]] && echo 1 || echo publication_defaults)"
    echo "baseline_sha256=$(sha256sum "$STUDY/baselines_fair4net.py" | awk '{print $1}')"
    echo "harness_sha256=$(sha256sum "$STUDY/baseline_harness_fair4net.py" | awk '{print $1}')"
} > "$tmp_contract"
if [[ -s "$CONTRACT" ]] && ! cmp -s "$CONTRACT" "$tmp_contract"; then
    echo "[mixed-baselines][ERROR] existing runner contract differs" >&2
    diff -u "$CONTRACT" "$tmp_contract" || true
    rm -f "$tmp_contract"
    exit 4
fi
mv -f "$tmp_contract" "$CONTRACT"

if [[ -f "$ODIR/.done" ]]; then
    python -u "$STUDY/validate_mixed_multinet.py" \
        --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role baselines --expected-N "$N"
    echo "[mixed-baselines] already complete and validated: $CASE"
    exit 0
fi

echo "[mixed-baselines] case=$CASE run_kind=$RUN_KIND job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$RUN_KIND" == smoke ]]; then
    python -u "$STUDY/baselines_fair4net.py" \
        --data "$DATA" --out_dir "$ODIR" --device cuda --seed 0 \
        --epochs_override 1 --allow_nonstandard_split --force
else
    python -u "$STUDY/baselines_fair4net.py" \
        --data "$DATA" --out_dir "$ODIR" --device cuda --seed 0
fi

touch "$ODIR/.done"
python -u "$STUDY/validate_mixed_multinet.py" \
    --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role baselines --expected-N "$N"
echo "[mixed-baselines] PASS case=$CASE end=$(date --iso-8601=seconds)"
