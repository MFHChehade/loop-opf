#!/bin/bash
# Build and audit one clean/line/generator mixed-contingency dataset.
# Submitter overrides the wall limit per case and run kind.
#SBATCH -J f4mx_ds
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -A YOUR_ALLOCATION

set -euo pipefail
: "${CASE:?Set CASE to one of the four supported networks}"
: "${CAMPAIGN_ID:?Set CAMPAIGN_ID}"
: "${RUN_KIND:?Set RUN_KIND=smoke or full}"
case "$CASE" in
    case300|case1354pegase|case2869pegase|case9241pegase) ;;
    *) echo "[mixed-dataset][ERROR] unsupported CASE=$CASE" >&2; exit 2 ;;
esac
case "$RUN_KIND" in
    smoke) N=120; CANDIDATES=8 ;;
    full) N=10000; CANDIDATES=32 ;;
    *) echo "[mixed-dataset][ERROR] RUN_KIND must be smoke or full" >&2; exit 2 ;;
esac

ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
OUTROOT="$ROOT/io/fair4net_mixed_contingency_${RUN_KIND}_v1/$CAMPAIGN_ID"
ODIR="$OUTROOT/$CASE/dataset"
DATA="$ODIR/dataset_mixed_N${N}_seed0.npz"
META="$ODIR/dataset_mixed_N${N}_seed0.meta.json"
AUDIT="$ODIR/dataset_audit.json"
CONTRACT="$ODIR/runner_contract.txt"

module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ODIR"
export DEQ_CASE="$CASE" PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

for path in \
    "$STUDY/make_mixed_contingency_dataset.py" \
    "$STUDY/make_line_contingency_dataset.py" \
    "$STUDY/audit_mixed_contingency_dataset.py" \
    "$STUDY/validate_mixed_multinet.py" \
    "$STUDY/deq_trainer_fair4net.py" \
    "$STUDY/dc_opf_pandapower_case300_9k.py"
do
    [[ -s "$path" ]] || { echo "[mixed-dataset][ERROR] missing $path" >&2; exit 3; }
done

python -m py_compile \
    "$STUDY/make_mixed_contingency_dataset.py" \
    "$STUDY/audit_mixed_contingency_dataset.py" \
    "$STUDY/validate_mixed_multinet.py"

tmp_contract=$(mktemp "$ODIR/.runner_contract.XXXXXX")
{
    echo "protocol=fair4net_mixed_n1_contingency_v1"
    echo "run_kind=$RUN_KIND"
    echo "case=$CASE"
    echo "N=$N"
    echo "seed=0"
    echo "line_candidates=$CANDIDATES"
    echo "generator_candidates=$CANDIDATES"
    echo "p_clean=0.25"
    echo "p_line=0.375"
    echo "p_generator=0.375"
    echo "progress_every=10"
    echo "ensure_candidate_coverage=true"
    echo "max_attempt_factor=3.0"
    echo "generator_sha256=$(sha256sum "$STUDY/make_mixed_contingency_dataset.py" | awk '{print $1}')"
    echo "audit_sha256=$(sha256sum "$STUDY/audit_mixed_contingency_dataset.py" | awk '{print $1}')"
    echo "trainer_sha256=$(sha256sum "$STUDY/deq_trainer_fair4net.py" | awk '{print $1}')"
} > "$tmp_contract"
if [[ -s "$CONTRACT" ]] && ! cmp -s "$CONTRACT" "$tmp_contract"; then
    echo "[mixed-dataset][ERROR] existing runner contract differs" >&2
    diff -u "$CONTRACT" "$tmp_contract" || true
    rm -f "$tmp_contract"
    exit 4
fi
mv -f "$tmp_contract" "$CONTRACT"

if [[ -f "$ODIR/.done" ]]; then
    python -u "$STUDY/validate_mixed_multinet.py" \
        --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role dataset --expected-N "$N"
    echo "[mixed-dataset] already complete and validated: $CASE"
    exit 0
fi

echo "[mixed-dataset] case=$CASE run_kind=$RUN_KIND N=$N job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[mixed-dataset] clean=0.25 line=0.375 generator=0.375 candidates=$CANDIDATES progress_every=10"
python -u "$STUDY/make_mixed_contingency_dataset.py" \
    --out "$DATA" --meta "$META" --N "$N" --seed 0 \
    --n-line-candidates "$CANDIDATES" --n-gen-candidates "$CANDIDATES" \
    --p-clean 0.25 --p-line 0.375 --p-gen 0.375 \
    --progress-every 10 --ensure-candidate-coverage --max-attempt-factor 3.0

python -u "$STUDY/audit_mixed_contingency_dataset.py" \
    --data "$DATA" --meta "$META" --out "$AUDIT" \
    --expected-N "$N" --require-candidate-coverage

touch "$ODIR/.done"
python -u "$STUDY/validate_mixed_multinet.py" \
    --root "$OUTROOT" --case "$CASE" --mode "$RUN_KIND" --role dataset --expected-N "$N"
sha256sum "$DATA" "$META" "$AUDIT" "$CONTRACT" > "$ODIR/source_manifest.sha256"
echo "[mixed-dataset] PASS case=$CASE end=$(date --iso-8601=seconds)"
