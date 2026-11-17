#!/bin/bash
#SBATCH -J topomix_full
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/topomix_full_%A_%a.out
#SBATCH -e logs/topomix_full_%A_%a.err

set -euo pipefail
ROOT=/path/to/workdir/DEQ_DC_OPF
DATA_ROOT="$ROOT/io/fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805"
CASES=(case300 case1354pegase case2869pegase case9241pegase)
METHODS=(admittance_dense dynamic_gnn)
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if (( TASK < 0 || TASK > 7 )); then echo "Invalid array index" >&2; exit 2; fi
DEQ_CASE=${CASES[$((TASK / 2))]}
METHOD=${METHODS[$((TASK % 2))]}
export DEQ_CASE PYTHONHASHSEED=0 OMP_NUM_THREADS=4
cd "$ROOT"
source /path/to/workdir/venv_deq/bin/activate
DATA="$DATA_ROOT/$DEQ_CASE/dataset/dataset_mixed_N10000_seed0.npz"
SMOKE="io/topology_conditioned_opf_v1/full_dataset_smoke/$DEQ_CASE/summary.json"
test -s "$DATA"
test -s "${DATA%/*}/dataset_audit.json"
test -s "$SMOKE"
python - "$DATA" "$SMOKE" "$METHOD" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

data, summary, method = map(Path, sys.argv[1:])
report = json.loads(summary.read_text(encoding="utf-8"))
assert report["protocol"] == "topology_conditioned_scenario_n1_dcopf_v1"
assert str(method) in report["results"]
assert math.isfinite(report["results"][str(method)]["metrics"]["rmse_MW"])
digest = hashlib.sha256()
with data.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == report["dataset_sha256"]
print("SMOKE_PROVENANCE_PASS", str(method), digest.hexdigest(), flush=True)
PY
python -u topology_conditioned_opf_v1/train.py \
  --data "$DATA" \
  --out-dir "io/topology_conditioned_opf_v1/full_100e/$DEQ_CASE" \
  --methods "$METHOD" --device cuda --epochs 100 \
  --batch-size 32 --lr 1e-3 --flow-weight 10 --seed 0 --resume
