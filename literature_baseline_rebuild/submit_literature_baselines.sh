#!/bin/bash
set -euo pipefail

ROOT=/path/to/workdir/DEQ_DC_OPF
CODE="$ROOT/literature_baseline_rebuild"
SLURM="$CODE/literature_baseline_job.slurm"
MODE=${1:-status}
RUN_TAG=${RUN_TAG:-literature_dc_v3_$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT="$ROOT/io/literature_baselines_dc_v3"
MANIFEST="$RESULT_ROOT/$RUN_TAG/submission.tsv"
CASES=${CASES:-"case300 case1354pegase case2869pegase case9241pegase"}
PROTOCOLS=${PROTOCOLS:-"clean mixed"}
METHODS_FILTER=${METHODS_FILTER:-"deepopf dc3 deeplde pdl canos"}

mkdir -p "$(dirname "$MANIFEST")"

submit_one() {
    local phase=$1 case=$2 protocol=$3 method=$4 epochs=$5 batch=$6 limit=$7 wall=$8
    local lr=$9 canos_h=${10} canos_s=${11} canos_updates=${12}
    local pdl_cap=${13} pdl_outer=${14} pdl_inner=${15}
    local name="lit_${phase}_${case#case}_${protocol:0:1}_${method}"
    name=${name:0:40}
    local raw jid
    raw=$(sbatch --parsable -J "$name" -t "$wall" \
        --export=ALL,CASE="$case",PROTOCOL="$protocol",METHOD="$method",EPOCHS="$epochs",BATCH_SIZE="$batch",SMOKE_LIMIT="$limit",RUN_TAG="$RUN_TAG",LR="$lr",CANOS_HIDDEN="$canos_h",CANOS_STEPS="$canos_s",CANOS_OPTIMIZER_STEPS="$canos_updates",PDL_HIDDEN_CAP="$pdl_cap",PDL_OUTER_ITERS="$pdl_outer",PDL_INNER_STEPS="$pdl_inner" \
        "$SLURM")
    # Vista's site wrapper can print a validation banner before the parsable ID.
    # Select the last numeric sbatch line instead of treating the entire stdout
    # stream as the job ID.
    jid=$(printf '%s\n' "$raw" | awk -F';' '/^[0-9]+(;.*)?$/ {id=$1} END {print id}')
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "[submit][ERROR] bad job id: $raw" >&2; exit 4; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$case" "$protocol" "$method" "$jid" "$epochs" "$batch" "$wall" \
        "$lr" "$canos_updates" "$pdl_outer" "$pdl_inner" >> "$MANIFEST"
    echo "[submit] $case $protocol $method -> $jid"
}

if [[ "$MODE" == smoke ]]; then
    printf 'phase\tcase\tprotocol\tmethod\tjob_id\tepochs\tbatch\ttime_limit\tlr\tcanos_updates\tpdl_outer\tpdl_inner\n' > "$MANIFEST"
    for case in $CASES; do
        case "$case" in
            case300) bs=8; wall=00:15:00 ;;
            case1354pegase) bs=8; wall=00:20:00 ;;
            case2869pegase) bs=4; wall=00:25:00 ;;
            case9241pegase) bs=2; wall=00:40:00 ;;
        esac
        for protocol in $PROTOCOLS; do
            # One sequential five-method smoke per case/protocol: 8 jobs total,
            # avoiding a 40-job queue burst while still exercising every path.
            submit_one smoke "$case" "$protocol" all 1 "$bs" 1 "$wall" 1e-3 8 1 0 256 20 0
        done
    done
    echo "[submit] smoke manifest=$MANIFEST"
elif [[ "$MODE" == full ]]; then
    if [[ "${CONFIRM_FULL:-0}" != 1 ]]; then
        echo "[submit][STOP] full mode is intentionally gated." >&2
        echo "Run the eight smokes, validate 40/40 outputs, review allocation, then set CONFIRM_FULL=1." >&2
        exit 5
    fi
    if [[ ! -s "$MANIFEST" ]]; then
        printf 'phase\tcase\tprotocol\tmethod\tjob_id\tepochs\tbatch\ttime_limit\tlr\tcanos_updates\tpdl_outer\tpdl_inner\n' > "$MANIFEST"
    fi
    for case in $CASES; do
        case "$case" in
            case300) dense_bs=200; dense_wall=03:00:00; canos_bs=16; canos_wall=08:00:00 ;;
            case1354pegase) dense_bs=128; dense_wall=05:00:00; canos_bs=8; canos_wall=12:00:00 ;;
            case2869pegase) dense_bs=64; dense_wall=08:00:00; canos_bs=4; canos_wall=18:00:00 ;;
            case9241pegase) dense_bs=32; dense_wall=14:00:00; canos_bs=2; canos_wall=24:00:00 ;;
            *) echo "[submit][ERROR] unsupported case=$case" >&2; exit 2 ;;
        esac
        for protocol in $PROTOCOLS; do
            [[ "$protocol" == clean || "$protocol" == mixed ]] || {
                echo "[submit][ERROR] unsupported protocol=$protocol" >&2; exit 2;
            }
            # Literature-derived budgets. PDL and CANOS are extremely costly
            # on one GH200; the explicit CONFIRM_FULL gate is therefore
            # mandatory and these limits must be reviewed after smoke timing.
            for method in $METHODS_FILTER; do
                result="$RESULT_ROOT/$RUN_TAG/$case/$protocol/$method/${method}_${protocol}_seed0.json"
                if [[ -s "$result" && "${RESUBMIT_COMPLETED:-0}" != 1 ]]; then
                    echo "[submit] skip completed $case $protocol $method"
                    continue
                fi
                case "$method" in
                    deepopf) submit_one full "$case" "$protocol" deepopf 200 64 0 "$dense_wall" 1e-4 16 2 0 1024 20 0 ;;
                    dc3) submit_one full "$case" "$protocol" dc3 1000 200 0 "$dense_wall" 1e-3 16 2 0 1024 20 0 ;;
                    deeplde) submit_one full "$case" "$protocol" deeplde 1000 200 0 "$dense_wall" 1e-3 16 2 0 1024 20 0 ;;
                    pdl) submit_one full "$case" "$protocol" pdl 1000 8 0 "$dense_wall" 1e-4 16 2 0 1024 20 2000 ;;
                    canos) submit_one full "$case" "$protocol" canos 1 "$canos_bs" 0 "$canos_wall" 1e-3 128 36 600000 1024 20 0 ;;
                    *) echo "[submit][ERROR] unsupported method=$method" >&2; exit 2 ;;
                esac
            done
        done
    done
    echo "[submit] full manifest=$MANIFEST"
elif [[ "$MODE" == status ]]; then
    latest=$(find "$RESULT_ROOT" -name submission.tsv -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
    echo "===== queue ====="
    squeue -u "$USER" -o '%.18i %.38j %.2t %.10M %.10l %.35R'
    if [[ -n "${latest:-}" && -s "$latest" ]]; then
        echo "===== latest manifest: $latest ====="
        cat "$latest"
        ids=$(tail -n +2 "$latest" | cut -f5 | paste -sd, -)
        if [[ -n "$ids" ]]; then
            sacct -X -j "$ids" --format=JobID,JobName%38,State,Elapsed,Timelimit,ExitCode
        fi
    fi
else
    echo "usage: bash submit_literature_baselines.sh {smoke|full|status}" >&2
    exit 2
fi
