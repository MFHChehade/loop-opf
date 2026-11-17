#!/bin/bash
#SBATCH -J scopf300_labels
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT=/path/to/workdir/phase_a_scopf_ieee300_full_v1
RUNDIR=/path/to/scratch/phase_a_scopf_ieee300_full_v1/generation/${SLURM_JOB_ID}
RESULTDIR=${PROJECT}/results/generation/${SLURM_JOB_ID}
NAME=ieee300_phase_a_K32_N10000_seed0
DATA=${RUNDIR}/${NAME}.npz
META=${RUNDIR}/${NAME}.meta.json
MANIFEST=${RUNDIR}/${NAME}.manifest.json
AUDIT=${RUNDIR}/${NAME}.audit.json
PROVENANCE=${RUNDIR}/${NAME}.provenance.json
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

module purge
module load gcc/13.2.0 cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
mkdir -p "$RUNDIR" "$RESULTDIR"
cd "$PROJECT"

python generate_phase_a_smoke.py \
  --out "$DATA" \
  --meta "$META" \
  --N 10000 \
  --seed 0 \
  --scale-low 0.9 \
  --scale-high 1.1 \
  --n-contingencies 32 \
  --progress-every 100 \
  --max-attempt-factor 3

python audit_phase_a_smoke.py \
  --data "$DATA" \
  --meta "$META" \
  --manifest "$MANIFEST" \
  --out "$AUDIT" \
  --provenance-out "$PROVENANCE" \
  --expected-N 10000 \
  --alternative-solver-samples 20 \
  --direct-progress-every 8

python - "$AUDIT" "$PROVENANCE" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
provenance = json.loads(Path(sys.argv[2]).read_text())
if audit.get("status") != "PASS" or not all(audit.get("gates", {}).values()):
    raise SystemExit(f"audit gates failed: {audit.get('gates')}")
if provenance.get("status") != "AUDITED_PASS":
    raise SystemExit(f"provenance failed: {provenance.get('status')}")
print("AUDITED_10000_LABELS_PASS")
PY

cp -a "${RUNDIR}/${NAME}."* "$RESULTDIR/"
(
  cd "$RESULTDIR"
  sha256sum \
    "${NAME}.npz" \
    "${NAME}.meta.json" \
    "${NAME}.manifest.json" \
    "${NAME}.audit.json" \
    "${NAME}.provenance.json" \
    > SHA256SUMS.txt
)

ARCHIVE=${PROJECT}/results/phase_a_ieee300_K32_N10000_labels-${SLURM_JOB_ID}.tgz
tar -C "${PROJECT}/results/generation" -czf "$ARCHIVE" "${SLURM_JOB_ID}"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
echo "PERSISTENT_LABEL_DIR=$RESULTDIR"
echo "LABEL_ARCHIVE=$ARCHIVE"
