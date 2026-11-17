"""Normalized error metrics for paper Table III (test samples without outages) and the RMSE scale for every table.

For one network (DEQ_CASE), on the clean test split of clean_table3_v1/<case>/dataset:
  * re-runs the saved best checkpoints of DeepOPF, DeepLDE, PDL (literature harness) and DC3 (strict-parity
    evaluator, with the authors' test-time correction), and the clean-trained Loop-OPF (deq_full_seed0);
  * GATE: each model's per-sample RMSE and worst flow violation must reproduce its saved per-sample record;
  * scores every prediction with the harness flow operator: worst branch overload / that branch's RATE_A
    (rated branches only, fmax < 1e5), per sample;
  * writes the RMSE scale p_bar = mean over test samples of (total demand / number of generators), for the clean
    test split and for the full mixed test split (Table IV), so RMSE can be reported as % of average generator output.
Nothing in the original campaign folders is modified.
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
case = args.case
os.environ["DEQ_CASE"] = case
ROOT = Path("/path/to/workdir/DEQ_DC_OPF")
IO = ROOT / "io"
CT = IO / "clean_table3_v1" / case
DATA = CT / "dataset" / "dataset_clean_from_mixed_seed0.npz"
MIXED = IO / "fair4net_mixed_contingency_full_v1/mixed_full_20260731_144805" / case / "dataset/dataset_mixed_N10000_seed0.npz"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "literature_baseline_rebuild"))
import run_literature_baselines as R  # noqa: E402

dev = "cuda"
# The baselines were trained/evaluated on the clean rows of the MIXED file (protocol "clean"); their per-sample
# records are keyed by the mixed-file row index, so prepare from the same file.
P = R.prepare(MIXED, dev, "clean")
# Run folders behind Table III (learning_opf_baselines_clean.csv); IEEE 300 lives in replay / clean_baselines_v1.
C300 = IO / "clean_baselines_v1/clean_baselines_full_case300_20260929_191010/case300"
RUN = {m: CT / f"{m}_full_seed0" for m in ("deepopf", "deeplde", "pdl", "dc3")}
if case == "case300":
    RUN.update(deepopf=CT / "deepopf_replay_seed0", deeplde=CT / "deeplde_replay_seed0", pdl=C300 / "pdl", dc3=CT / "dc3_replay_seed0")
test = P.idx["test"]
fmax = torch.as_tensor(P.static["fmax"], dtype=torch.float32, device=P.device)
rated = fmax < 1e5
print(f"[{case}] clean test rows {len(test)}, rated lines {int(rated.sum())}", flush=True)

def rel_score(pred, idx_np):
    idx = R.tensor_index(idx_np, P.device)
    v = P.op.flow_violation_vec(pred.float(), P.pd_bus[idx], P.cont_line[idx])
    rel = torch.where(rated[None, :], v / fmax[None, :], torch.zeros_like(v))
    rmse = torch.sqrt(((pred.float() - P.pg[idx]) ** 2).mean(1))
    return v.max(1).values.double().cpu().numpy(), rel.max(1).values.double().cpu().numpy(), rmse.double().cpu().numpy()

def gate(name, ab, rm, ref_ab, ref_rm):
    d1 = float(np.max(np.abs(ab - ref_ab) / np.maximum(1.0, np.abs(ref_ab))))
    d2 = float(np.max(np.abs(rm - ref_rm) / np.maximum(1.0, np.abs(ref_rm))))
    ok = d1 <= 1e-3 and d2 <= 1e-3
    print(f"[gate] {name} flow_maxreldiff={d1:.3g} rmse_maxreldiff={d2:.3g} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok, d1, d2

res, gates = {}, {}
# ---------------- literature harness baselines (DeepOPF, DeepLDE, PDL)
for m in ("deepopf", "deeplde", "pdl"):
    d = RUN[m]; stem = f"{m}_clean_seed0"
    sig = torch.load(d / f"{stem}_training_state.pt", map_location="cpu", weights_only=False)["signature"]
    if m == "deepopf":
        model = R.DeepOPFNet(P.X.shape[1], P.ngen, hidden_base=int(sig["deepopf_hidden_base"]))
        fn = lambda b, mm=None: R.equality_decode(P, model(P.X[b]), b)
    elif m == "deeplde":
        model = R.EqualityEmbeddedMLP(P.X.shape[1], P.ngen)
        model.register_buffer("learned_inequality_dual", torch.zeros(P.nbranch + 2 * P.ngen))
        fn = lambda b: R.equality_decode(P, model(P.X[b]), b)
    else:
        hidden = max(64, min(int(sig["pdl_hidden_cap"]), int(1.5 * P.X.shape[1])))
        model = R.PDLBundle(P.X.shape[1], P.ngen, P.nbranch + 2 * P.ngen, hidden)
        fn = lambda b: R.pdl_decode(P, torch.sigmoid(model.primal(P.X[b])), b)
    state = torch.load(d / f"{stem}.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True); model.to(P.device).eval()
    with torch.no_grad():
        pred = R.predict_batched(P, fn, test, int(sig["batch_size"]))
    ab, rl, rm = rel_score(pred, test)
    ref = {int(r["source_index"]): r for r in csv.DictReader(open(d / f"{stem}_per_sample.csv"))}
    src = P.source_index[test]
    gates[m] = gate(m, ab, rm, np.array([float(ref[int(s)]["flow_violation_MW"]) for s in src]),
                    np.array([float(ref[int(s)]["rmse_MW"]) for s in src]))
    res[m] = dict(abs=ab, rel=rl, rmse=rm)

# ---------------- DC3 (authors' engine + test-time correction), captured from the strict-parity evaluator
sys.path.insert(0, str(ROOT / "dc3_strict_parity_v2"))
import evaluate_fair4net_physics as E  # noqa: E402
cap = {}
_orig_helpers = E.load_fair4net_helpers
def _helpers():
    h = _orig_helpers(); orig = h.score_predictions
    def sp(Pp, pred, idx_np):
        cap.setdefault("pred", pred.detach().clone()); cap.setdefault("P", Pp); cap.setdefault("idx", idx_np)
        return orig(Pp, pred, idx_np)
    h.score_predictions = sp; return h
E.load_fair4net_helpers = _helpers
sys.argv = ["evaluate_fair4net_physics.py", "--data", str(MIXED), "--checkpoint", str(RUN["dc3"] / "solver_net.dict"),
            "--protocol", "clean", "--case", case, "--out", str(out / f"{case}_dc3_reaudit.json")]
E.main()
P2 = cap["P"]
idx = R.tensor_index(cap["idx"], P2.device)
v = P2.op.flow_violation_vec(cap["pred"].float(), P2.pd_bus[idx], P2.cont_line[idx])
fm2 = torch.as_tensor(P2.static["fmax"], dtype=torch.float32, device=P2.device)
rl = torch.where((fm2 < 1e5)[None, :], v / fm2[None, :], torch.zeros_like(v)).max(1).values.double().cpu().numpy()
ab = v.max(1).values.double().cpu().numpy()
rm = torch.sqrt(((cap["pred"].float() - P2.pg[idx]) ** 2).mean(1)).double().cpu().numpy()
ref = json.load(open(RUN["dc3"] / "physics_audit.json"))["metrics_after_official_dc3_correction"]["all"]
ok = abs(ab.mean() - ref["viol_MW"]) <= 1e-3 * max(1, ref["viol_MW"]) and abs(float(np.sqrt((rm ** 2).mean())) - ref["rmse_MW"]) <= 1e-3 * max(1, ref["rmse_MW"])
print(f"[gate] dc3 viol new={ab.mean():.6g} ref={ref['viol_MW']:.6g} rmse(pooled) new={np.sqrt((rm**2).mean()):.6g} ref={ref['rmse_MW']:.6g} -> {'PASS' if ok else 'FAIL'}", flush=True)
gates["dc3"] = (bool(ok), None, None)
res["dc3"] = dict(abs=ab, rel=rl, rmse=rm)

# ---------------- Loop-OPF trained without outages (clean_table3_v1 deq_full_seed0), current shared trainer
sys.path.insert(0, str(ROOT / "fair4net_l1_study"))
import deq_trainer_fair4net as T  # noqa: E402
TR_SHA = "3180377b25eb36a3ecf49b2b9a219639cfddb68c147aa26c8e25c63d16c90acb"
got = hashlib.sha256(open(ROOT / "fair4net_l1_study/deq_trainer_fair4net.py", "rb").read()).hexdigest()
assert got == TR_SHA, got
LD = CT / "deq_full_seed0"; LS = f"deq_{case}_clean_full_seed0"
summ = json.load(open(LD / f"{LS}_summary.json"))   # <LS>.json is the per-epoch history (a list)
c = summ["config"]
BATCH = {"case300": 16, "case1354pegase": 8, "case2869pegase": 8, "case9241pegase": 4}[case]
cfg = T.TrainCfg(device="cuda", batch_size=BATCH, picard_iters=int(c["picard_iters"]), anderson_m=int(c["anderson_m"]),
                 anderson_lam=float(c["anderson_lam"]), anderson_max_iter=int(c["anderson_max_iter"]), anderson_tol=float(c["anderson_tol"]))
ds = T.DCOPFDataset(str(DATA)); tidx = np.where(ds.split == 2)[0]
net = T.build_base_net(); ppc = T.build_ppc_from_net(net); snp = T.extract_static_mats(ppc)
snp["lodf"] = ds.lodf if ds.lodf is not None else T.extract_lodf(ppc, snp)
static = T.to_torch_static(snp, cfg)
ck = torch.load(LD / f"{LS}.pt", map_location="cuda", weights_only=False)
steps = T.LearnableSteps(nK=float(ck["nK"]), s_max=cfg.s_max, rho_init=0.9, s_init=0.0).to("cuda"); steps.load_state_dict(ck["steps_state"])
op = T.PDHGOperatorLearnable(static, steps).to("cuda")
deq = T.DEQFixedPoint(f=op, dim=static["ngen"] + 1 + 2 * static["nbranch"], picard_iters=cfg.picard_iters,
                      fwd_kwargs=dict(m=cfg.anderson_m, lam=cfg.anderson_lam, max_iter=cfg.anderson_max_iter, tol=cfg.anderson_tol, beta=cfg.anderson_beta),
                      bwd_kwargs=dict(m=cfg.b_anderson_m, lam=cfg.b_anderson_lam, max_iter=cfg.b_anderson_max_iter, tol=cfg.b_anderson_tol, beta=cfg.b_anderson_beta)).to("cuda")
deq.eval(); op.eval(); ng = int(static["ngen"])
preds = []
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(tidx), BATCH):
        b = [ds[int(i)] for i in tidx[s:s + BATCH]]
        d = torch.stack([x["pd_bus"] for x in b]).to("cuda", cfg.dtype)
        cl = torch.stack([torch.as_tensor(x["cont_line_idx"]) for x in b]).to("cuda", torch.long)
        cg = torch.stack([torch.as_tensor(x["cont_gen_idx"]) for x in b]).to("cuda", torch.long)
        preds.append(deq(d, cg, cl)[:, :ng].float())
pred = torch.cat(preds)
print(f"[{case}] Loop-OPF inference {time.time() - t0:.1f}s", flush=True)
assert len(tidx) == len(test), (len(tidx), len(test))
# the DEQ dataset and the harness share the clean subset in the same row order; check labels agree
lab = torch.stack([ds[int(i)]["pg"] for i in tidx]).to(P.device).float()
assert torch.allclose(lab, P.pg[R.tensor_index(test, P.device)].float(), atol=1e-3), "row order mismatch DEQ vs harness"
ab, rl, rm = rel_score(pred, test)
ref = list(csv.DictReader(open(LD / f"{LS}_eval_vs_gt.csv")))
gates["Loop-OPF"] = gate("Loop-OPF", ab, rm, np.array([float(r["viol_max_pred"]) for r in ref]), np.array([float(r["rmse_MW"]) for r in ref]))
res["Loop-OPF"] = dict(abs=ab, rel=rl, rmse=rm)

# ---------------- RMSE scale: average generator output = total demand / n_gen, mean over test rows
def pbar(path):
    z = np.load(path, allow_pickle=True)
    split = z["split"]
    tot = z["pg"][split == 2].sum(1)        # total OPF generation = total served demand (incl. shunt load)
    return float(np.mean(tot / ng)), int((split == 2).sum())
pb_clean, n_c = pbar(DATA); pb_mixed, n_m = pbar(MIXED)

summary = {"case": case, "n_gen": ng, "n_rated_lines": int(rated.sum()),
           "pbar_clean_MW": pb_clean, "pbar_clean_n": n_c, "pbar_mixed_MW": pb_mixed, "pbar_mixed_n": n_m,
           "gates": {k: bool(v[0]) for k, v in gates.items()}, "methods": {}}
for k, r in res.items():
    summary["methods"][k] = {"worst_abs_mean_MW": float(r["abs"].mean()), "worst_rel_mean_pct": float(100 * r["rel"].mean()),
                             "rmse_mean_MW": float(r["rmse"].mean()), "rmse_pooled_MW": float(np.sqrt((r["rmse"] ** 2).mean())),
                             "rmse_pct_of_pbar_clean": float(100 * r["rmse"].mean() / pb_clean)}
np.savez(out / f"{case}_table3_norm.npz", **{f"{k}_{q}": v for k, r in res.items() for q, v in r.items()})
json.dump(summary, open(out / f"{case}_table3_norm_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=1), flush=True)
if not all(v[0] for v in gates.values()):
    raise SystemExit("GATE FAILED: do not use these numbers")
