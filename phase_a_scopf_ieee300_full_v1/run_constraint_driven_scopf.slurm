#!/bin/bash
#SBATCH -J giraudK32
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
PROJECT=/path/to/workdir/phase_a_scopf_ieee300_full_v1
DATA=${PROJECT}/results/generation/904082/ieee300_phase_a_K32_N10000_seed0.npz
MODE=${MODE:?set MODE to smoke or full}
cd "${PROJECT}"
mkdir -p logs "results/constraint_driven_scopf/${MODE}"
module purge
module load gcc/13.2.0 cuda/12.6
source /path/to/workdir/venv_deq/bin/activate
test "$(sha256sum "${DATA}" | awk '{print $1}')" = 901a6f563eb4fc92c1692b1c85d25d8a21ae511f7170499a7370a1687517d863
test -f "${DATA%.npz}.provenance.json"
test -f "${DATA%.npz}.audit.json"
python -c 'import torch; assert torch.cuda.is_available()'
EXTRA=(--project)
if [[ "${MODE}" == smoke ]]; then
  EXTRA+=(--smoke --epochs 1)
elif [[ "${MODE}" == full ]]; then
  EXTRA+=(--epochs 300)
else
  echo "Invalid MODE=${MODE}" >&2
  exit 2
fi
python train_constraint_driven_scopf.py \
  --data "${DATA}" \
  --output-dir "${PROJECT}/results/constraint_driven_scopf/${MODE}" \
  --device cuda --batch-size 64 --eval-batch-size 256 \
  --hidden 64 --lr 1e-4 --security-weight 100 --seed 0 \
  "${EXTRA[@]}"
python - "${PROJECT}/results/constraint_driven_scopf/${MODE}/results.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["status"] == "PASS"
assert report["raw_metrics"]["sample_count"] == (20 if report["split_counts"]["test"] == 20 else 2000)
assert report["projected_metrics"]["complete_scopf_feasibility_rate"] == 1.0
print("RESULTS_VALIDATED", report["protocol"])
PY
