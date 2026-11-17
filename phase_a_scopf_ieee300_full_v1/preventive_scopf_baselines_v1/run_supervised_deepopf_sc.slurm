#!/bin/bash
#SBATCH -J dopsck32
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o logs/dopsck32_%j.out
#SBATCH -e logs/dopsck32_%j.err

set -euo pipefail
PROJECT=/path/to/workdir/phase_a_scopf_ieee300_full_v1
DATA="$PROJECT/results/generation/904082/ieee300_phase_a_K32_N10000_seed0.npz"
MODE=${MODE:?set MODE to smoke or full}
cd "$PROJECT"
source /path/to/workdir/venv_deq/bin/activate
export OMP_NUM_THREADS=4 PYTHONHASHSEED=0
test "$(sha256sum "$DATA" | awk '{print $1}')" = 901a6f563eb4fc92c1692b1c85d25d8a21ae511f7170499a7370a1687517d863
test -s "${DATA%.npz}.provenance.json"
test -s "${DATA%.npz}.audit.json"
python -c 'import torch; assert torch.cuda.is_available()'
if [[ "$MODE" == smoke ]]; then
  EPOCHS=1
  EXTRA=(--smoke)
elif [[ "$MODE" == full ]]; then
  EPOCHS=300
  EXTRA=()
else
  echo "Invalid MODE" >&2
  exit 2
fi
python -u preventive_scopf_baselines_v1/train_supervised_deepopf_sc.py \
  --data "$DATA" \
  --output-dir "$PROJECT/results/supervised_deepopf_sc/$MODE" \
  --device cuda --epochs "$EPOCHS" --batch-size 64 \
  --eval-batch-size 256 --hidden 64 --lr 1e-4 \
  --security-weight 100 --seed 0 --project "${EXTRA[@]}"
python - "$PROJECT/results/supervised_deepopf_sc/$MODE/results.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["status"] == "PASS"
assert report["raw_metrics"]["sample_count"] == (20 if report["split_counts"]["test"] == 20 else 2000)
assert report["projected_metrics"]["complete_scopf_feasibility_rate"] == 1.0
assert report["checkpoint_reload_max_abs_mw"] <= 1e-3
print("RESULTS_VALIDATED", report["protocol"])
PY
