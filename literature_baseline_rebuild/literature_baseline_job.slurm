#!/bin/bash
#SBATCH -J litdc
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.out
#SBATCH -e /path/to/workdir/DEQ_DC_OPF/logs/%x_%j.err

set -euo pipefail

ROOT=/path/to/workdir/DEQ_DC_OPF
CODE="$ROOT/literature_baseline_rebuild"
: "${CASE:?CASE is required}"
: "${PROTOCOL:?PROTOCOL is required}"
: "${METHOD:?METHOD is required}"

EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-200}
LR=${LR:-1e-3}
CANOS_HIDDEN=${CANOS_HIDDEN:-16}
CANOS_STEPS=${CANOS_STEPS:-2}
CANOS_OPTIMIZER_STEPS=${CANOS_OPTIMIZER_STEPS:-0}
PDL_HIDDEN_CAP=${PDL_HIDDEN_CAP:-1024}
PDL_OUTER_ITERS=${PDL_OUTER_ITERS:-20}
PDL_INNER_STEPS=${PDL_INNER_STEPS:-0}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-50}
CHECKPOINT_STEP_INTERVAL=${CHECKPOINT_STEP_INTERVAL:-5000}
VALIDATION_STEP_INTERVAL=${VALIDATION_STEP_INTERVAL:-10000}
RUN_TAG=${RUN_TAG:-literature_dc_v3}
SMOKE_LIMIT=${SMOKE_LIMIT:-0}

if [[ "$PROTOCOL" == clean ]]; then
    DATA="$ROOT/io/fair4net_l1_v1/$CASE/N10000_seed0/random/dataset_random_gsaware.npz"
elif [[ "$PROTOCOL" == mixed ]]; then
    DATA="$ROOT/io/fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805/$CASE/dataset/dataset_mixed_N10000_seed0.npz"
else
    echo "[litdc][ERROR] unsupported protocol=$PROTOCOL" >&2
    exit 2
fi

OUT="$ROOT/io/literature_baselines_dc_v3/$RUN_TAG/$CASE/$PROTOCOL/$METHOD"
methods=("$METHOD")
if [[ "$METHOD" == all ]]; then
    methods=(deepopf dc3 deeplde pdl canos)
fi
for path in "$DATA" "$CODE/run_literature_baselines.py" "$ROOT/fair4net_l1_study/baseline_harness_fair4net.py"; do
    [[ -s "$path" ]] || { echo "[litdc][ERROR] missing $path" >&2; exit 3; }
done

module load cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
cd "$ROOT"
mkdir -p logs "$OUT"

export DEQ_CASE="$CASE"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

extra=()
if [[ "$SMOKE_LIMIT" == 1 ]]; then
    extra+=(--smoke-limit --force)
fi

echo "[litdc] job=${SLURM_JOB_ID:-local} case=$CASE protocol=$PROTOCOL method=$METHOD"
echo "[litdc] epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR canos=${CANOS_STEPS}x${CANOS_HIDDEN}/updates=$CANOS_OPTIMIZER_STEPS smoke=$SMOKE_LIMIT"
echo "[litdc] pdl=outer$PDL_OUTER_ITERS/inner$PDL_INNER_STEPS/hidden_cap$PDL_HIDDEN_CAP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u "$CODE/run_literature_baselines.py" \
    --data "$DATA" \
    --out-dir "$OUT" \
    --protocol "$PROTOCOL" \
    --methods "${methods[@]}" \
    --device cuda \
    --seed 0 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --canos-hidden "$CANOS_HIDDEN" \
    --canos-steps "$CANOS_STEPS" \
    --canos-optimizer-steps "$CANOS_OPTIMIZER_STEPS" \
    --pdl-hidden-cap "$PDL_HIDDEN_CAP" \
    --pdl-outer-iters "$PDL_OUTER_ITERS" \
    --pdl-inner-steps "$PDL_INNER_STEPS" \
    --validation-interval "$VALIDATION_INTERVAL" \
    --checkpoint-step-interval "$CHECKPOINT_STEP_INTERVAL" \
    --validation-step-interval "$VALIDATION_STEP_INTERVAL" \
    --resume \
    "${extra[@]}"

echo "[litdc] PASS"
