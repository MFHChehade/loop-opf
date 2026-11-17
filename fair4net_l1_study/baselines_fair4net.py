#!/usr/bin/env python3
"""
Run the seven local DC-OPF baseline implementations on one explicit corrected
dataset. The network is selected via ``DEQ_CASE`` before this module is
imported. All methods use the same GS-aware physics, split, and metric
definitions as ``deq_trainer_fair4net.py``.

Usage:
  DEQ_CASE=case300 python baselines_fair4net.py \
    --data io/fair4net_l1_v1/case300/N10000_seed0/random/dataset_random_gsaware.npz \
    --out_dir io/fair4net_l1_v2_phaseaware/runs/case300/baselines \
    --device cuda --seed 0
"""
from __future__ import annotations
import os, sys, json, time, argparse, pathlib, hashlib, csv
import numpy as np
import torch
import torch.nn as nn
import baseline_harness_fair4net as H

_HERE = pathlib.Path(__file__).resolve().parent
SIZES = [1000, 10000, 20000]
SEEDS = [0, 1, 2]
SPLITS = ["random", "disjoint"]
BASELINES = ["mlp", "deepopf", "dc3", "deeplde", "pdl", "reserve", "gnn"]
PROTOCOL = "fair4net_l1_v2_phase_aware"
FLOW_MODEL_CONTRACT = "f=PTDF@(Pg-PD-GS)+phase_shift_offset"


class MLP(nn.Module):
    def __init__(self, i, o, h=(256, 256)):
        super().__init__()
        dims = [i, *h]; L = []
        for a, b in zip(dims[:-1], dims[1:]):
            L += [nn.Linear(a, b), nn.ReLU()]
        L += [nn.Linear(dims[-1], o)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def data_path(case, N, seed, split):
    if case == "case9241pegase":
        return _HERE / "io" / "pegase9k_contingency_ablation" / f"N{N}_seed{seed}" / f"dataset_{split}.npz"
    return _HERE / "io" / "multinet_ablation" / case / f"N{N}_seed{seed}" / f"dataset_{split}.npz"


def prep(op, static, cfg, path, device):
    """Load one dataset; reuse the pre-built operator/static for this network."""
    d = H.load_gencont(str(path))
    H.validate_contract(d, static, require_metadata=True)
    if d.get("lodf") is not None:
        expected = (int(static["nbranch"]), int(static["nbranch"]))
        if d["lodf"].shape != expected or not np.all(np.isfinite(d["lodf"])):
            raise ValueError(f"dataset lodf is invalid: shape={d['lodf'].shape}, expected={expected}")
        op.LODF = torch.as_tensor(d["lodf"], dtype=op.PTDF.dtype, device=device)
        print("[fair-baselines] using finite outage-specific LODF columns embedded in dataset")
    N = d["pd_bus"].shape[0]
    pd_bus = torch.from_numpy(d["pd_bus"]).to(device)
    d_eff = op.effective_demand(pd_bus)
    pg = torch.from_numpy(d["pg"]).to(device) if d["pg"] is not None else None
    cont = d["cont"] if d["cont"] is not None else np.full(N, -1, np.int64)
    cont_line = d["cont_line"] if d["cont_line"] is not None else np.full(N, -1, np.int64)
    cont_line_t = torch.from_numpy(cont_line).to(device=device, dtype=torch.long)
    has_line_contingencies = bool(np.any(cont_line >= 0))
    line_features = torch.zeros(N, int(static["nbranch"]), dtype=op.PTDF.dtype, device=device)
    valid_line = cont_line_t >= 0
    if torch.any(valid_line):
        # Continuous physics encoding: selected standard LODF column.  This is
        # meaningful for validation/test outage identities never seen in train,
        # unlike an untrained one-hot branch ID.
        line_features[valid_line] = op.LODF[:, cont_line_t[valid_line]].t()
    pmax_eff = H.effective_pmax(static, cont, device)
    ngen = int(static["ngen"])
    pmin = torch.as_tensor(static["pmin"], dtype=torch.float32, device=device)[None, :].expand(N, -1)
    # Honor the dataset's precomputed split EXACTLY as the DEQ trainer does:
    # integer encoding 0=train, 1=val, 2=test (critical for the disjoint OOD test).
    sp = np.asarray(d["split"]) if d["split"] is not None else None
    if sp is not None and sp.dtype.kind in "iu":
        idx = {"train": np.where(sp == 0)[0], "val": np.where(sp == 1)[0], "test": np.where(sp == 2)[0]}
    elif sp is not None and (set(np.unique(sp.astype(str))) & {"train", "val", "test"}):
        ss = sp.astype(str)
        idx = {"train": np.where(ss == "train")[0], "val": np.where(ss == "val")[0], "test": np.where(ss == "test")[0]}
    else:
        rng = np.random.default_rng(0); perm = rng.permutation(N)
        idx = {"train": perm[:int(.7*N)], "val": perm[int(.7*N):int(.8*N)], "test": perm[int(.8*N):]}
    X = torch.cat([d_eff, pmax_eff, line_features], 1) if has_line_contingencies else torch.cat([d_eff, pmax_eff], 1)
    tr = idx["train"]
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True).clamp_min(1e-6)
    return dict(op=op, static=static, ngen=ngen, nbus=int(static["nbus"]), device=device,
                pd_bus=pd_bus, d_eff=d_eff, pg=pg, pmin=pmin, pmax_eff=pmax_eff,
                cont_gen=torch.from_numpy(cont).to(device=device, dtype=torch.long),
                cont_line=cont_line_t, line_features=line_features,
                has_line_contingencies=has_line_contingencies,
                input_dim=int(X.shape[1]), X=(X - mu) / sd, idx=idx)


def _line(P, idx):
    """Return per-sample outage-line indices on the operator device."""
    if P["cont_line"] is None:
        return None
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx, device=P["cont_line"].device)
    return P["cont_line"][idx]


def _gen(P, idx):
    """Return per-sample outage-generator indices on the operator device."""
    if P["cont_gen"] is None:
        return None
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx, device=P["cont_gen"].device)
    return P["cont_gen"][idx]


def _fit(P, params, loss_fn, predict_fn, epochs, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    tr = torch.as_tensor(P["idx"]["train"], device=P["device"])
    t0 = time.time()
    for ep in range(epochs):
        b = tr[torch.randperm(len(tr), device=P["device"])]
        loss = loss_fn(b)
        opt.zero_grad(); loss.backward(); opt.step()
    te = P["idx"]["test"]
    with torch.no_grad():
        m = H.score(P["op"], predict_fn(te), P["pd_bus"][te], P["pg"][te] if P["pg"] is not None else None,
                    cont_line_idx=_line(P, te), cont_gen_idx=_gen(P, te))
    m["train_time_sec"] = time.time() - t0
    return m


def _feas(P, p, idx):
    p = H.box_clamp(p, P["pmin"][idx], P["pmax_eff"][idx])
    return H.balance_repair(p, P["d_eff"][idx], P["pmin"][idx], P["pmax_eff"][idx])


def _stats(P):
    tr = P["idx"]["train"]
    return P["pg"][tr].mean(0, keepdim=True), P["pg"][tr].std(0, keepdim=True).clamp_min(1e-3)


# ── the seven baselines (validated logic; each returns a metrics dict) ─────────
def run_mlp(P, epochs, seed, lr=1e-3):
    mu, sd = _stats(P); m = MLP(P["input_dim"], P["ngen"]).to(P["device"])
    loss = lambda b: torch.mean((m(P["X"][b]) - (P["pg"][b] - mu) / sd) ** 2)
    pred = lambda i: _feas(P, m(P["X"][i]) * sd + mu, i)
    return _fit(P, m.parameters(), loss, pred, epochs, lr, seed)


def run_deepopf(P, epochs, seed, lr=1e-3):
    mu, sd = _stats(P); m = MLP(P["input_dim"], P["ngen"]).to(P["device"])

    def recover(p, idx):
        pmx = P["pmax_eff"][idx]; dsum = P["d_eff"][idx].sum(1)
        slack = pmx.argmax(1); rows = torch.arange(p.shape[0], device=p.device)
        mask = torch.zeros_like(p); mask[rows, slack] = 1.0
        p_o = p * (1 - mask); p = p_o + mask * (dsum - p_o.sum(1))[:, None]
        return _feas(P, p, idx)
    loss = lambda b: torch.mean((m(P["X"][b]) - (P["pg"][b] - mu) / sd) ** 2)
    pred = lambda i: recover(m(P["X"][i]) * sd + mu, i)
    return _fit(P, m.parameters(), loss, pred, epochs, lr, seed)


def run_dc3(P, epochs, seed, lr=1e-3, Tcorr=15, gamma=5e-4):
    mu, sd = _stats(P); op = P["op"]; m = MLP(P["input_dim"], P["ngen"]).to(P["device"])

    def flow_pen(p, idx):
        return (op.flow_violation_vec(p, P["pd_bus"][idx], _line(P, idx)) ** 2).mean()

    def correct(p, idx):
        for _ in range(Tcorr):
            with torch.enable_grad():
                pin = p.detach().requires_grad_(True)
                g = torch.autograd.grad(flow_pen(pin, idx), pin)[0]
            p = _feas(P, p - gamma * g, idx)
        return p
    loss = lambda b: torch.mean((m(P["X"][b]) - (P["pg"][b] - mu) / sd) ** 2)
    pred = lambda i: correct(_feas(P, m(P["X"][i]) * sd + mu, i), i)
    return _fit(P, m.parameters(), loss, pred, epochs, lr, seed)


def run_deeplde(P, epochs, seed, lr=1e-3, eta=0.05):
    mu, sd = _stats(P); op = P["op"]; dev = P["device"]
    nbr = int(P["static"]["nbranch"])
    fmax = torch.as_tensor(P["static"]["fmax"], dtype=torch.float32, device=dev)
    dual = torch.zeros(nbr, device=dev)
    m = MLP(P["input_dim"], P["ngen"]).to(dev)
    nfv = lambda p, idx: op.flow_violation_vec(p, P["pd_bus"][idx], _line(P, idx)) / fmax[None, :].clamp_min(1e-6)

    def loss(b):
        p = _feas(P, m(P["X"][b]) * sd + mu, b)
        mse = torch.mean((p - P["pg"][b]) ** 2) / float(sd.mean() ** 2)
        return mse + (dual[None, :] * nfv(p, b)).sum(1).mean()

    def pred(i):
        return _feas(P, m(P["X"][i]) * sd + mu, i)

    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = torch.as_tensor(P["idx"]["train"], device=dev); t0 = time.time()
    for ep in range(epochs):
        b = tr[torch.randperm(len(tr), device=dev)]
        L = loss(b); opt.zero_grad(); L.backward(); opt.step()
        if ep % 5 == 0:
            with torch.no_grad():
                p = _feas(P, m(P["X"][b]) * sd + mu, b)
                dual.copy_(torch.clamp(dual + eta * nfv(p, b).mean(0), min=0.0))
    te = P["idx"]["test"]
    with torch.no_grad():
        mm = H.score(op, pred(te), P["pd_bus"][te], P["pg"][te], cont_line_idx=_line(P, te), cont_gen_idx=_gen(P, te)); mm["train_time_sec"] = time.time() - t0
    return mm


def run_pdl(P, epochs, seed, lr=1e-3, rho=1.0):
    op = P["op"]; dev = P["device"]
    mu, sd = _stats(P)
    nbr = int(P["static"]["nbranch"])
    fmax = torch.as_tensor(P["static"]["fmax"], dtype=torch.float32, device=dev)
    primal = MLP(P["input_dim"], P["ngen"]).to(dev)
    dual = MLP(P["input_dim"], nbr).to(dev)
    nfv = lambda p, idx: op.flow_violation_vec(p, P["pd_bus"][idx], _line(P, idx)) / fmax[None, :].clamp_min(1e-6)
    cost = lambda p, idx: op.gen_cost(p).mean() / P["d_eff"][idx].sum(1).mean()
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(list(primal.parameters()) + list(dual.parameters()), lr=lr)
    tr = torch.as_tensor(P["idx"]["train"], device=dev); t0 = time.time()
    for ep in range(epochs):
        b = tr[torch.randperm(len(tr), device=dev)]
        p = _feas(P, primal(P["X"][b]) * sd + mu, b)
        v = nfv(p, b); lam = torch.nn.functional.softplus(dual(P["X"][b]))
        Lp = cost(p, b) + (lam.detach() * v).sum(1).mean() + 0.5 * rho * (v ** 2).sum(1).mean()
        tgt = (lam.detach() + rho * v.detach()).clamp_min(0.0)
        L = Lp + torch.mean((lam - tgt) ** 2)
        opt.zero_grad(); L.backward(); opt.step()
    te = P["idx"]["test"]
    with torch.no_grad():
        p = _feas(P, primal(P["X"][te]) * sd + mu, te)
        mm = H.score(op, p, P["pd_bus"][te], P["pg"][te], cont_line_idx=_line(P, te), cont_gen_idx=_gen(P, te)); mm["train_time_sec"] = time.time() - t0
    return mm


def run_reserve(P, epochs, seed, lr=1e-3, iters=30):
    mu, sd = _stats(P); m = MLP(P["input_dim"], P["ngen"]).to(P["device"])

    def response(p, idx):
        pmn = P["pmin"][idx]; pmx = P["pmax_eff"][idx]; dsum = P["d_eff"][idx].sum(1)
        r = (pmx - pmn).clamp_min(1e-3)
        lo = torch.full((p.shape[0],), -1e4, device=p.device); hi = torch.full((p.shape[0],), 1e4, device=p.device)
        for _ in range(iters):
            s = 0.5 * (lo + hi)
            tot = H.box_clamp(p + s[:, None] * r, pmn, pmx).sum(1)
            hi = torch.where(tot > dsum, s, hi); lo = torch.where(tot > dsum, lo, s)
        s = 0.5 * (lo + hi)
        return _feas(P, H.box_clamp(p + s[:, None] * r, pmn, pmx), idx)
    loss = lambda b: torch.mean((m(P["X"][b]) - (P["pg"][b] - mu) / sd) ** 2)
    pred = lambda i: response(m(P["X"][i]) * sd + mu, i)
    return _fit(P, m.parameters(), loss, pred, epochs, lr, seed)


_GRAPH_CACHE = {}


def run_gnn(P, epochs, seed, lr=1e-3, hidden=64, layers=3):
    dev = P["device"]; nbus = P["nbus"]
    mu, sd = _stats(P)
    bog = torch.as_tensor(P["static"]["bus_of_gen"], dtype=torch.long, device=dev)
    if nbus not in _GRAPH_CACHE:
        f, t = H.build_graph()
        ii = torch.as_tensor(np.concatenate([f, t, np.arange(nbus)]), device=dev)
        jj = torch.as_tensor(np.concatenate([t, f, np.arange(nbus)]), device=dev)
        deg = torch.zeros(nbus, device=dev).index_add_(0, ii, torch.ones_like(ii, dtype=torch.float32))
        dinv = deg.pow(-0.5); dinv[torch.isinf(dinv)] = 0.0
        val = dinv[ii] * dinv[jj]
        with torch.sparse.check_sparse_tensor_invariants():
            A = torch.sparse_coo_tensor(
                torch.stack([ii, jj]), val, (nbus, nbus)
            ).coalesce()
        _GRAPH_CACHE[nbus] = A
    A = _GRAPH_CACHE[nbus]

    def spmm(h):                                   # A @ h for h (B,nbus,hid) via 2D sparse.mm
        B, n, hd = h.shape
        x = h.permute(1, 0, 2).reshape(n, B * hd)
        y = torch.sparse.mm(A, x)
        return y.reshape(n, B, hd).permute(1, 0, 2)

    class GNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Linear(2, hidden)
            s.cont = nn.Linear(int(P["static"]["nbranch"]), hidden, bias=False)
            s.convs = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
            s.ro = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(s, node, pmx, pmn, line_features):
            h = torch.relu(s.enc(node) + s.cont(line_features)[:, None, :])
            for c in s.convs:
                h = torch.relu(c(torch.cat([h, spmm(h)], -1)))
            hg = h[:, bog, :]
            return s.ro(torch.cat([hg, pmx[..., None], pmn[..., None]], -1)).squeeze(-1)

    N = P["pd_bus"].shape[0]
    cap = torch.zeros(N, nbus, device=dev).index_add_(1, bog, P["pmax_eff"])
    node = torch.stack([P["d_eff"], cap], -1)
    trn = P["idx"]["train"]
    nm = node[trn].reshape(-1, 2).mean(0); ns = node[trn].reshape(-1, 2).std(0).clamp_min(1e-6)
    node = (node - nm) / ns
    m = GNN().to(dev)
    # mini-batch so the (B, nbus, hidden) activation stays bounded (full-batch OOMs on big nets)
    bs = max(16, min(256, 1_500_000 // nbus))
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    trn = torch.as_tensor(P["idx"]["train"], device=dev)
    t0 = time.time()
    for ep in range(epochs):
        perm = trn[torch.randperm(len(trn), device=dev)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            z = m(node[b], P["pmax_eff"][b], P["pmin"][b], P["line_features"][b])
            loss = torch.mean((z - (P["pg"][b] - mu) / sd) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    te = torch.as_tensor(P["idx"]["test"], device=dev)
    with torch.no_grad():
        preds = [_feas(P, m(node[b], P["pmax_eff"][b], P["pmin"][b], P["line_features"][b]) * sd + mu, b)
                 for b in (te[i:i + bs] for i in range(0, len(te), bs))]
        p_hat = torch.cat(preds, 0)
        idx = P["idx"]["test"]
        mm = H.score(P["op"], p_hat, P["pd_bus"][idx], P["pg"][idx], cont_line_idx=_line(P, idx), cont_gen_idx=_gen(P, idx))
    mm["train_time_sec"] = time.time() - t0
    return mm


RUNNERS = {"mlp": run_mlp, "deepopf": run_deepopf, "dc3": run_dc3, "deeplde": run_deeplde,
           "pdl": run_pdl, "reserve": run_reserve, "gnn": run_gnn}
EPOCHS = {"mlp": 500, "deepopf": 500, "dc3": 500, "deeplde": 500, "pdl": 800, "reserve": 500, "gnn": 200}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    """Match the array digest recorded by prepare_and_audit_dataset.py."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.view(np.uint8))
    return h.hexdigest()


def _write_json_atomic(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _write_summary(out_dir: pathlib.Path, rows: list[dict]) -> None:
    _write_json_atomic(out_dir / "baseline_summary.json", {"rows": rows})
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    tmp = out_dir / "baseline_summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_dir / "baseline_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out_dir", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=BASELINES, choices=BASELINES)
    ap.add_argument(
        "--epochs_override",
        type=int,
        default=None,
        help="testing only: replace each method's publication-run epoch count",
    )
    ap.add_argument(
        "--allow_nonstandard_split",
        action="store_true",
        help="smoke-test only: accept explicit split counts other than 7000/1000/2000",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    case = os.environ.get("DEQ_CASE")
    if not case:
        raise RuntimeError("Set DEQ_CASE before launching this script.")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    print(f"[fair-baselines] case={case} data={args.data} device={args.device}")
    op, static, cfg = H.build_operator(args.device)
    print(
        f"[fair-baselines] operator nbus={static['nbus']} ngen={static['ngen']} "
        f"nbranch={static['nbranch']} GS={float(np.sum(static['gs_bus'])):.6f} MW"
    )
    P = prep(op, static, cfg, args.data, args.device)
    split_counts = {name: int(len(P["idx"][name])) for name in ("train", "val", "test")}
    if not args.allow_nonstandard_split and split_counts != {"train": 7000, "val": 1000, "test": 2000}:
        raise ValueError(f"Expected split counts 7000/1000/2000, got {split_counts}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_sha = _sha256(args.data)
    flow_offset = np.asarray(static["flow_offset"])
    flow_offset_sha = _sha256_array(flow_offset)
    flow_offset_abs_max = (
        float(np.max(np.abs(flow_offset))) if flow_offset.size else 0.0
    )
    flow_offset_nonzero_count = int(np.count_nonzero(np.abs(flow_offset) > 1e-12))
    phase_shift_nonzero_branches = int(static["phase_shift_nonzero_count"])
    flow_model_contract = str(static["flow_model_contract"])
    if flow_model_contract != FLOW_MODEL_CONTRACT:
        raise ValueError(
            "Unexpected flow-model contract: "
            f"{flow_model_contract!r} != {FLOW_MODEL_CONTRACT!r}"
        )
    rows: list[dict] = []
    has_line = bool(torch.any(P["cont_line"] >= 0).item())
    has_gen = bool(torch.any(P["cont_gen"] >= 0).item())
    split_label = "mixed_disjoint" if has_line and has_gen else ("line_disjoint" if has_line else "random")
    for name in args.only:
        outf = args.out_dir / f"{name}.json"
        if outf.exists() and not args.force:
            row = json.loads(outf.read_text(encoding="utf-8"))
            if (
                row.get("protocol") == PROTOCOL
                and row.get("flow_offset_sha256") == flow_offset_sha
                and row.get("dataset_sha256") == data_sha
                and row.get("split") == split_label
            ):
                rows.append(row)
                print(f"[fair-baselines] skip complete {name}: {outf}")
                continue
            print(
                f"[fair-baselines] stale {name}; rerunning "
                f"(protocol={row.get('protocol')!r}, "
                f"flow_offset_sha256={row.get('flow_offset_sha256')!r})"
            )

        # Seed before model construction as well as inside each optimizer loop.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        method_epochs = EPOCHS[name] if args.epochs_override is None else args.epochs_override
        if method_epochs < 1:
            raise ValueError("--epochs_override must be at least 1.")
        print(f"[fair-baselines] start {name} epochs={method_epochs}")
        m = RUNNERS[name](P, method_epochs, args.seed)
        required_metrics = (
            "bal_abs_MW",
            "bal_max_MW",
            "viol_MW",
            "viol_p95_MW",
            "viol_branch_mean_MW",
            "viol_max_MW",
            "rmse_MW",
            "cost_rel_gap_abs",
        )
        bad = [key for key in required_metrics if not np.isfinite(m.get(key, np.nan))]
        if bad:
            raise FloatingPointError(f"{name} produced non-finite metrics: {bad}")
        m.update(
            dict(
                case=case,
                N=int(P["pd_bus"].shape[0]),
                seed=args.seed,
                split=split_label,
                baseline=name,
                protocol=PROTOCOL,
                scenario_protocol=(
                    "fair4net_mixed_n1_contingency_v1" if has_line and has_gen else split_label
                ),
                demand_contract="effective_demand=pd_bus+gs_bus",
                flow_model_contract=flow_model_contract,
                flow_offset_sha256=flow_offset_sha,
                flow_offset_abs_max_MW=flow_offset_abs_max,
                flow_offset_nonzero_count=flow_offset_nonzero_count,
                phase_shift_nonzero_branches=phase_shift_nonzero_branches,
                flow_metric_contract=(
                    "viol_MW=mean sample worst branch; "
                    "viol_branch_mean_MW=mean sample-branch"
                ),
                contingency_encoding=(
                    "lodf_column+pmax_zero"
                    if P["has_line_contingencies"] and has_gen
                    else ("lodf_column" if P["has_line_contingencies"] else ("pmax_zero" if has_gen else "none"))
                ),
                dataset_path=str(args.data),
                dataset_sha256=data_sha,
                epochs=method_epochs,
                train_samples=split_counts["train"],
                val_samples=split_counts["val"],
                test_samples=split_counts["test"],
            )
        )
        _write_json_atomic(outf, m)
        rows.append(m)
        _write_summary(args.out_dir, rows)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"[fair-baselines] done {name:8s} "
            f"gap={100*m.get('cost_rel_gap_abs', float('nan')):.6g}% "
            f"worst_mean={m['viol_MW']:.6g} MW "
            f"branch_mean={m['viol_branch_mean_MW']:.6g} MW "
            f"max={m['viol_max_MW']:.6g} MW "
            f"rmse={m.get('rmse_MW', float('nan')):.6g} MW "
            f"time={m['train_time_sec']:.1f}s"
        )

    _write_summary(args.out_dir, rows)
    if set(args.only) == set(BASELINES):
        (args.out_dir / ".done").write_text("complete\n", encoding="utf-8")
    print(f"[fair-baselines] complete methods={len(rows)} out={args.out_dir}")


if __name__ == "__main__":
    main()
