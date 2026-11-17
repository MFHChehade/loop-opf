"""Relative flow violations for paper Table IV (mixed-contingency test, 2,000 samples).

Re-runs each saved Loop-OPF checkpoint with the exact trainer of the original
campaign (sha 5c3899...), and re-scores the topology-conditioned baselines from
their saved test predictions with the SAME flow operator. For each sample it
stores the worst absolute violation (MW) and the worst violation relative to
the line rating (violation / RATE_A).

Reproduction gate: Loop-OPF per-sample worst violation and RMSE must match the
campaign's saved eval_vs_gt.csv; baseline worst violation must match their
result.json means. Any mismatch aborts before writing results.
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.environ["DEQ_CASE"] = args.case
sys.path.insert(0, args.code)
import deq_trainer_fair4net as T  # noqa: E402

TRAINER_SHA = "5c3899ca0181647605e244800317c3cf4f622ae88966227d02d239a97ca07626"
got = hashlib.sha256(open(os.path.join(args.code, "deq_trainer_fair4net.py"), "rb").read()).hexdigest()
assert got == TRAINER_SHA, got

W = Path("/path/to/workdir/DEQ_DC_OPF/io")
CAMP = W / "fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805" / args.case
DATA = CAMP / "dataset/dataset_mixed_N10000_seed0.npz"
RUN = CAMP / "deq_l1_scalar_seed0"
STEM = f"deq_{args.case}_mixed_l1_scalar_e30"
BASE = W / "topology_conditioned_opf_v1/full_100e" / args.case
BATCH = {"case300": 16, "case1354pegase": 8, "case2869pegase": 8, "case9241pegase": 4}[args.case]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# contract of the original run
contract = dict(l.strip().split("=", 1) for l in open(RUN / "runner_contract.txt") if "=" in l)
assert contract["trainer_sha256"] == TRAINER_SHA
assert hashlib.sha256(open(DATA, "rb").read()).hexdigest() == contract["dataset_sha256"]

dev = torch.device("cuda")
cfg = T.TrainCfg(device="cuda", batch_size=BATCH, picard_iters=int(contract["picard_iters"]), anderson_m=int(contract["anderson_m"]),
                 anderson_lam=float(contract["anderson_lam"]), anderson_max_iter=int(contract["anderson_max_iter"]),
                 anderson_tol=float(contract["anderson_tol"]))
ds = T.DCOPFDataset(str(DATA))
assert ds.split is not None
test_idx = np.where(ds.split == 2)[0]
assert len(test_idx) == 2000

net = T.build_base_net(); ppc = T.build_ppc_from_net(net); static_np = T.extract_static_mats(ppc)
static_np["lodf"] = ds.lodf if ds.lodf is not None else T.extract_lodf(ppc, static_np)
static = T.to_torch_static(static_np, cfg)
ck = torch.load(RUN / f"{STEM}.pt", map_location=dev, weights_only=False)
steps = T.LearnableSteps(nK=float(ck["nK"]), s_max=cfg.s_max, rho_init=0.9, s_init=0.0).to(dev)
steps.load_state_dict(ck["steps_state"])
op = T.PDHGOperatorLearnable(static, steps).to(dev)
deq = T.DEQFixedPoint(f=op, dim=static["ngen"] + 1 + 2 * static["nbranch"], picard_iters=cfg.picard_iters,
                      fwd_kwargs=dict(m=cfg.anderson_m, lam=cfg.anderson_lam, max_iter=cfg.anderson_max_iter, tol=cfg.anderson_tol, beta=cfg.anderson_beta),
                      bwd_kwargs=dict(m=cfg.b_anderson_m, lam=cfg.b_anderson_lam, max_iter=cfg.b_anderson_max_iter, tol=cfg.b_anderson_tol, beta=cfg.b_anderson_beta)).to(dev)
deq.eval(); op.eval()
fmax = op.fmax.detach()
real = fmax < 1e5          # unrated branches carry a large placeholder limit
ngen = int(static["ngen"])

@torch.no_grad()
def score(p, d_bus, cl):
    v = op.flow_violation_vec(p, d_bus, cl)                  # (B, nbranch) MW
    rel = torch.where(real[None, :], v / fmax[None, :], torch.zeros_like(v))
    return v.max(dim=1).values, rel.max(dim=1).values

def rows_for(idx):
    b = [ds[int(i)] for i in idx]
    d = torch.stack([x["pd_bus"] for x in b]).to(dev, cfg.dtype)
    pt = torch.stack([x["pg"] for x in b]).to(dev, cfg.dtype)
    cl = torch.stack([torch.as_tensor(x["cont_line_idx"]) for x in b]).to(dev, torch.long)
    cg = torch.stack([torch.as_tensor(x["cont_gen_idx"]) for x in b]).to(dev, torch.long)
    return d, pt, cl, cg

t0 = time.time()
res = {k: [] for k in ("loop_abs", "loop_rel", "loop_rmse", "lab_abs", "lab_rel")}
with torch.no_grad():
    for s in range(0, 2000, BATCH):
        d, pt, cl, cg = rows_for(test_idx[s:s + BATCH])
        p = deq(d, cg, cl)[:, :ngen]
        a, r = score(p, d, cl); res["loop_abs"].append(a); res["loop_rel"].append(r)
        res["loop_rmse"].append(torch.sqrt(((p - pt) ** 2).mean(dim=1)))
        a, r = score(pt, d, cl); res["lab_abs"].append(a); res["lab_rel"].append(r)
res = {k: torch.cat(v).double().cpu().numpy() for k, v in res.items()}
print(f"[{args.case}] Loop-OPF inference {time.time() - t0:.1f}s", flush=True)

# reproduction gate against the campaign's own per-sample CSV
ref = list(csv.DictReader(open(RUN / f"{STEM}_eval_vs_gt.csv")))
rv = np.array([float(x["viol_max_pred"]) for x in ref]); rr = np.array([float(x["rmse_MW"]) for x in ref])
gate = dict(viol_maxdiff=float(np.max(np.abs(rv - res["loop_abs"]))), viol_mean_ref=float(rv.mean()), viol_mean_new=float(res["loop_abs"].mean()),
            rmse_mean_ref=float(rr.mean()), rmse_mean_new=float(res["loop_rmse"].mean()))
print("[gate]", json.dumps(gate), flush=True)
ok_loop = abs(gate["viol_mean_new"] - gate["viol_mean_ref"]) <= 1e-3 * max(1.0, gate["viol_mean_ref"]) and \
          abs(gate["rmse_mean_new"] - gate["rmse_mean_ref"]) <= 1e-3 * max(1.0, gate["rmse_mean_ref"])

# baselines from their saved predictions, same operator
base = {}
for m in ("admittance_dense", "dynamic_gnn"):
    z = np.load(BASE / m / "test_predictions.npz")
    rj = json.load(open(BASE / m / "result.json"))
    ref_mean = float(rj["metrics"]["viol_worst_mean_MW"]) if "metrics" in rj else float("nan")
    rows = z["dataset_row_index"]
    A, Rl = [], []
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            d, pt, cl, cg = rows_for(rows[s:s + 64])
            assert np.allclose(pt.cpu().numpy(), z["pg_label"][s:s + 64], atol=1e-3), "label order mismatch"
            p = torch.as_tensor(z["pg_pred"][s:s + 64], device=dev, dtype=cfg.dtype)
            a, r = score(p, d, cl); A.append(a); Rl.append(r)
    A = torch.cat(A).double().cpu().numpy(); Rl = torch.cat(Rl).double().cpu().numpy()
    base[m] = dict(abs=A, rel=Rl, ref_mean=ref_mean)
    print(f"[gate] {m} viol mean new={A.mean():.4f} ref={ref_mean:.4f}", flush=True)
ok_base = all(abs(b["abs"].mean() - b["ref_mean"]) <= 1e-2 * max(1.0, b["ref_mean"]) for b in base.values())

np.savez(out / f"{args.case}_relviol.npz", fmax=fmax.cpu().numpy(), **res,
         **{f"{m}_{k}": base[m][k] for m in base for k in ("abs", "rel")})
TOL_REL = [1e-4, 1e-3, 1e-2]
TOL_ABS = [1e-3, 1e-2, 1e-1, 1.0]
summ = {"case": args.case, "gate_loop": gate, "gate_ok_loop": bool(ok_loop), "gate_ok_baselines": bool(ok_base),
        "n_rated_lines": int(real.sum().item()), "methods": {}}
for name, a, r in [("label", res["lab_abs"], res["lab_rel"]), ("Loop-OPF", res["loop_abs"], res["loop_rel"])] + \
                  [(m, base[m]["abs"], base[m]["rel"]) for m in base]:
    summ["methods"][name] = {"worst_abs_mean_MW": float(a.mean()), "worst_rel_mean_pct": float(100 * r.mean()),
                             **{f"sat_abs_{t:g}MW_pct": float(100 * np.mean(a <= t)) for t in TOL_ABS},
                             **{f"sat_rel_{100*t:g}pct_pct": float(100 * np.mean(r <= t)) for t in TOL_REL}}
json.dump(summ, open(out / f"{args.case}_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
if not (ok_loop and ok_base):
    raise SystemExit("REPRODUCTION GATE FAILED: do not use these numbers")
