#!/bin/bash
# Submit exactly 12 production jobs: dataset -> {DEQ, baselines} for four cases.

set -euo pipefail

ROOT=/path/to/workdir/DEQ_DC_OPF
STUDY="$ROOT/fair4net_l1_study"
BASE_OUT="$ROOT/io/fair4net_mixed_contingency_full_v1"
ATTESTATION="$ROOT/mixed_contingency_local_smoke_attestation.json"
CAMPAIGN_ID="mixed_full_$(date +%Y%m%d_%H%M%S)"
OUTROOT="$BASE_OUT/$CAMPAIGN_ID"
MANIFEST="$OUTROOT/submission.tsv"
CONTRACT="$OUTROOT/campaign_contract.txt"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTROOT"

python - "$ATTESTATION" "$STUDY" <<'PY'
import hashlib, json, pathlib, sys
attestation = pathlib.Path(sys.argv[1])
study = pathlib.Path(sys.argv[2])
if not attestation.is_file():
    raise SystemExit(f"missing local smoke attestation: {attestation}")
value = json.loads(attestation.read_text())
expected_cases = {"case300", "case1354pegase", "case2869pegase", "case9241pegase"}
if value.get("status") != "PASS" or set(value.get("cases", {})) != expected_cases:
    raise SystemExit("local smoke attestation does not PASS all four cases")
for case, row in value["cases"].items():
    if row.get("status") != "PASS":
        raise SystemExit(f"local smoke did not pass for {case}")
    if any(int(v) != 0 for v in row.get("split_isolation", {}).values()):
        raise SystemExit(f"split isolation failed for {case}")
for name, expected in value.get("source_sha256", {}).items():
    path = study / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source changed after local smoke: {name}: {actual} != {expected}")
print("[submit-mixed-full] local four-case smoke attestation and source hashes PASS")
PY

submit() {
    local output jid
    output=$(sbatch "$@")
    printf '%s\n' "$output" >&2
    jid=$(printf '%s\n' "$output" | awk '
        /Submitted batch job [0-9]+/ {id=$NF}
        /^[0-9]+([;].*)?$/ {split($1,a,";"); id=a[1]}
        END {print id}
    ')
    if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
        echo "[submit-mixed-full][ERROR] could not parse job id" >&2
        exit 4
    fi
    printf '%s\n' "$jid"
}

{
    echo "protocol=fair4net_mixed_n1_contingency_v1"
    echo "campaign=$CAMPAIGN_ID"
    echo "N=10000"
    echo "seed=0"
    echo "epochs=30"
    echo "line_candidates=32"
    echo "generator_candidates=32"
    echo "p_clean=0.25"
    echo "p_line=0.375"
    echo "p_generator=0.375"
    echo "split_isolation=line_and_generator_identity_disjoint_plus_exact_demand_disjoint"
    echo "smoke_attestation_sha256=$(sha256sum "$ATTESTATION" | awk '{print $1}')"
    echo "dataset_code_sha256=$(sha256sum "$STUDY/make_mixed_contingency_dataset.py" | awk '{print $1}')"
    echo "audit_code_sha256=$(sha256sum "$STUDY/audit_mixed_contingency_dataset.py" | awk '{print $1}')"
    echo "trainer_code_sha256=$(sha256sum "$STUDY/deq_trainer_fair4net.py" | awk '{print $1}')"
    echo "baseline_code_sha256=$(sha256sum "$STUDY/baselines_fair4net.py" | awk '{print $1}')"
} > "$CONTRACT"

printf 'campaign\trole\tcase\tjob_id\tdependency\ttime_limit\n' > "$MANIFEST"

# Limits are based on the completed line-contingency N=10k campaigns.  Mixed
# samples do not apply both outages simultaneously, so per-row physics remains
# one N-1 OPF. Requested walltime is failure margin; SUs follow actual runtime.
for item in \
    "case300 300 01:00:00 16:00:00 00:20:00" \
    "case1354pegase 1354 02:00:00 1-00:00:00 00:30:00" \
    "case2869pegase 2869 04:00:00 1-00:00:00 00:45:00" \
    "case9241pegase 9241 1-06:00:00 2-00:00:00 01:30:00"
do
    read -r case_name short_name dataset_time deq_time baseline_time <<< "$item"
    export_spec="ALL,CASE=$case_name,CAMPAIGN_ID=$CAMPAIGN_ID,RUN_KIND=full"

    dataset_job=$(submit \
        -J "f4mx_${short_name}_ds" -t "$dataset_time" \
        --export="$export_spec" \
        "$STUDY/fair4net_mixed_dataset.slurm")

    deq_job=$(submit \
        -J "f4mx_${short_name}_deq" -t "$deq_time" \
        --dependency="afterok:$dataset_job" \
        --export="$export_spec" \
        "$STUDY/fair4net_mixed_deq.slurm")

    baseline_job=$(submit \
        -J "f4mx_${short_name}_base" -t "$baseline_time" \
        --dependency="afterok:$dataset_job" \
        --export="$export_spec" \
        "$STUDY/fair4net_mixed_baselines.slurm")

    printf '%s\tdataset\t%s\t%s\t-\t%s\n' \
        "$CAMPAIGN_ID" "$case_name" "$dataset_job" "$dataset_time" >> "$MANIFEST"
    printf '%s\tdeq\t%s\t%s\tafterok:%s\t%s\n' \
        "$CAMPAIGN_ID" "$case_name" "$deq_job" "$dataset_job" "$deq_time" >> "$MANIFEST"
    printf '%s\tbaselines\t%s\t%s\tafterok:%s\t%s\n' \
        "$CAMPAIGN_ID" "$case_name" "$baseline_job" "$dataset_job" "$baseline_time" >> "$MANIFEST"
    echo "[submit-mixed-full] case=$case_name dataset=$dataset_job deq=$deq_job baselines=$baseline_job"
done

job_count=$(( $(wc -l < "$MANIFEST") - 1 ))
if [[ "$job_count" -ne 12 ]]; then
    echo "[submit-mixed-full][ERROR] manifest does not contain exactly 12 jobs" >&2
    exit 5
fi
printf '%s\n' "$CAMPAIGN_ID" > "$BASE_OUT/latest_campaign.txt"
echo "[submit-mixed-full] manifest=$MANIFEST"
echo "[submit-mixed-full] campaign=$CAMPAIGN_ID"
squeue -u "$USER" -o '%.18i %.28j %.2t %.10M %.10l %.40R'
