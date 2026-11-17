"""Train the GNN baseline of Table III on the samples WITHOUT outages only (clean_table3_v1 dataset, same split as
DeepOPF/DC3/DeepLDE/PDL/Loop-OPF), with the original runner baselines_fair4net.py --only gnn (seed 0, its
publication epoch count), and score it on the same clean test samples.

Outputs per network: RMSE (pooled, MW), mean worst flow violation (MW), Flow (%) = mean over test samples of the
worst branch overload / that branch's RATE_A (rated branches, fmax < 1e5), balance error (MW), cost gap (%).
"""
import json, os, sys, pathlib
import numpy as np

case = sys.argv[1]
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
os.environ["DEQ_CASE"] = case
ROOT = pathlib.Path("/path/to/workdir/DEQ_DC_OPF")
DATA = ROOT / "io/clean_table3_v1" / case / "dataset/dataset_clean_from_mixed_seed0.npz"
sys.path.insert(0, str(ROOT / "fair4net_l1_study"))
import torch  # noqa: E402
import baseline_harness_fair4net as H  # noqa: E402
import baselines_fair4net as B  # noqa: E402

cap = {}
_score = H.score


def score(op, p_hat, d_bus, p_star=None, eps=1e-9, cont_line_idx=None, cont_gen_idx=None):
    m = _score(op, p_hat, d_bus, p_star, eps, cont_line_idx, cont_gen_idx)
    cap.update(p_hat=p_hat.detach(), p_star=p_star.detach(), v=op.flow_violation_vec(p_hat, d_bus, cont_line_idx).detach())
    return m


H.score = score
sys.argv = ["baselines_fair4net.py", "--data", str(DATA), "--out_dir", str(out / "run"), "--device", "cuda",
            "--seed", "0", "--only", "gnn", "--force", "--allow_nonstandard_split"]
B.main()

_, static, _ = H.build_operator("cuda")
v = cap["v"]
fmax = torch.as_tensor(np.asarray(static["fmax"]), dtype=v.dtype, device=v.device)
rated = fmax < 1e5
worst = v.max(1).values
rel = torch.where(rated[None, :], v / fmax[None, :], torch.zeros_like(v)).max(1).values
m = json.load(open(out / "run/gnn.json"))
res = dict(case=case, n_test=int(v.shape[0]), n_rated_lines=int(rated.sum()), train=m["train_samples"],
           val=m["val_samples"], test=m["test_samples"], epochs=m["epochs"],
           rmse_pooled_MW=float(torch.sqrt(torch.mean((cap["p_hat"] - cap["p_star"]) ** 2))), rmse_MW_json=m["rmse_MW"],
           flow_MW=float(worst.mean()), flow_MW_json=m["viol_MW"], flow_rel_pct=100 * float(rel.mean()),
           balance_abs_MW=m["bal_abs_MW"], cost_gap_pct=100 * m["cost_rel_gap_abs"], outage_rows_in_test=m.get("line_samples", 0) + m.get("generator_samples", 0))
json.dump(res, open(out / f"{case}_gnn_clean.json", "w"), indent=2)
np.savez_compressed(out / f"{case}_gnn_clean_pred.npz", p_hat=cap["p_hat"].cpu().numpy(), worst_MW=worst.cpu().numpy(),
                    worst_rel=rel.cpu().numpy())
print("[gnn-clean]", json.dumps(res), flush=True)
