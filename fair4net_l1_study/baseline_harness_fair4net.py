#!/usr/bin/env python3
"""
Shared harness for baseline comparison against the solver-informed DEQ.

Reuses the trainer's physics operators (gen_cost, flow_violation_vec) so every
baseline is scored with the IDENTICAL metrics as the DEQ. Network is selected via
the DEQ_CASE environment variable (same mechanism as the trainer).

Public API:
  build_operator(device)       -> (op, static_np, cfg)
  load_gencont(npz_path)       -> dict(pd_bus, pg, cont, split, GS metadata)
  validate_contract(...)       -> fail-fast data/model contract validation
  effective_pmax(static, cont) -> (B, ngen) pmax with outaged gen zeroed
  score(op, p_hat, d_bus, p_star) -> harmonized DEQ/baseline metrics
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import deq_trainer_fair4net as tr  # noqa: E402


def build_operator(device: str = "cpu"):
    """Build the DC-OPF physics operator for the current DEQ_CASE network."""
    net = tr.build_base_net()
    ppc = tr.build_ppc_from_net(net)
    static_np = tr.extract_static_mats(ppc)
    # A line-contingency dataset supplies audited outage-specific columns.
    # Avoid constructing the full base LODF here: islanding columns can emit
    # divide-by-zero warnings, and they must never substitute for the dataset's
    # validated contingency physics.
    static_np["lodf"] = np.zeros(
        (int(static_np["nbranch"]), int(static_np["nbranch"])), dtype=np.float32
    )
    cfg = tr.TrainCfg(device=device)
    st = tr.to_torch_static(static_np, cfg)
    steps = tr.LearnableSteps(nK=1.0, s_max=cfg.s_max)  # unused by scoring, required by ctor
    op = tr.PDHGOperatorLearnable(st, steps).to(device)
    return op, static_np, cfg


def build_graph():
    """Branch endpoints (from_bus, to_bus) as 0-based internal bus indices, for GNN baselines."""
    net = tr.build_base_net()
    ppc = tr.build_ppc_from_net(net)
    try:
        from pypower.idx_brch import F_BUS, T_BUS
    except Exception:
        from pandapower.pypower.idx_brch import F_BUS, T_BUS
    br = ppc["branch"]
    f = np.real(br[:, F_BUS]).astype(np.int64)
    t = np.real(br[:, T_BUS]).astype(np.int64)
    return f, t


def load_gencont(npz_path: str) -> dict:
    with np.load(npz_path, allow_pickle=True) as z:
        return dict(
            pd_bus=z["pd_bus"].astype(np.float32),
            pg=z["pg"].astype(np.float32) if "pg" in z.files else None,
            cont=z["cont_gen_idx"].astype(np.int64) if "cont_gen_idx" in z.files else None,
            cont_line=z["cont_line_idx"].astype(np.int64) if "cont_line_idx" in z.files else None,
            lodf=z["lodf"].astype(np.float32) if "lodf" in z.files else None,
            split=np.array(z["split"], copy=True) if "split" in z.files else None,
            gs_bus=z["gs_bus"].astype(np.float32) if "gs_bus" in z.files else None,
            pd_bus_includes_gs=bool(int(np.asarray(z["pd_bus_includes_gs"]).item()))
            if "pd_bus_includes_gs" in z.files else None,
            contract_version=str(np.asarray(z["contract_version"]).item())
            if "contract_version" in z.files else None,
        )


def validate_contract(d: dict, static_np: dict, *, require_metadata: bool = True) -> None:
    """Validate the explicit PD + GS contract and network dimensions."""
    pd_bus = d["pd_bus"]
    if pd_bus.ndim != 2 or pd_bus.shape[1] != int(static_np["nbus"]):
        raise ValueError(
            f"pd_bus shape {pd_bus.shape} does not match nbus={int(static_np['nbus'])}."
        )
    if not np.all(np.isfinite(pd_bus)):
        raise ValueError("pd_bus contains non-finite values.")
    if d["pg"] is not None:
        if d["pg"].shape != (pd_bus.shape[0], int(static_np["ngen"])):
            raise ValueError(
                f"pg shape {d['pg'].shape} does not match "
                f"(N, ngen)=({pd_bus.shape[0]}, {int(static_np['ngen'])})."
            )
        if not np.all(np.isfinite(d["pg"])):
            raise ValueError("pg contains non-finite values.")

    if require_metadata and d["gs_bus"] is None:
        raise ValueError("Corrected dataset is missing required gs_bus metadata.")
    if d["pd_bus_includes_gs"] not in (False, None):
        raise ValueError("pd_bus_includes_gs must be 0; pd_bus stores conventional PD only.")
    if require_metadata and d["pd_bus_includes_gs"] is None:
        raise ValueError("Corrected dataset is missing pd_bus_includes_gs metadata.")
    if d["gs_bus"] is not None:
        gs = np.asarray(d["gs_bus"], dtype=np.float32)
        expected = np.asarray(static_np["gs_bus"], dtype=np.float32)
        if gs.shape != expected.shape or not np.allclose(gs, expected, rtol=0.0, atol=1e-6):
            err = float(np.max(np.abs(gs - expected))) if gs.shape == expected.shape else float("inf")
            raise ValueError(f"Dataset GS does not match selected network; max_abs_error={err:.6g} MW.")

    split = d["split"]
    if split is None:
        raise ValueError("Fair-study dataset must contain its fixed train/val/test split.")
    split = np.asarray(split)
    if split.shape != (pd_bus.shape[0],):
        raise ValueError(f"split shape {split.shape} does not match N={pd_bus.shape[0]}.")
    vals = set(np.unique(split.astype(np.int64)).tolist())
    if not vals.issubset({0, 1, 2}):
        raise ValueError(f"split contains unexpected values: {sorted(vals)}")

    cont = d["cont"]
    if cont is not None:
        if cont.shape != (pd_bus.shape[0],):
            raise ValueError(f"cont_gen_idx shape {cont.shape} does not match N={pd_bus.shape[0]}.")
        if np.any((cont < -1) | (cont >= int(static_np["ngen"]))):
            raise ValueError("cont_gen_idx contains an out-of-range generator index.")

    cont_line = d.get("cont_line")
    if cont_line is not None:
        if cont_line.shape != (pd_bus.shape[0],):
            raise ValueError(f"cont_line_idx shape {cont_line.shape} does not match N={pd_bus.shape[0]}.")
        if np.any((cont_line < -1) | (cont_line >= int(static_np["nbranch"]))):
            raise ValueError("cont_line_idx contains an out-of-range branch index.")
        if np.any(cont_line >= 0) and d.get("lodf") is None:
            raise ValueError("line-contingency samples require an embedded audited LODF matrix.")


def effective_pmax(static_np: dict, cont, device: str = "cpu",
                   dtype=torch.float32) -> torch.Tensor:
    """Per-sample effective upper bound: pmax with the outaged generator zeroed.
    This is the contingency-encoding input augmentation used by every baseline."""
    pmax = torch.as_tensor(static_np["pmax"], dtype=dtype, device=device)
    cont_t = torch.as_tensor(cont, dtype=torch.long, device=device)
    B = cont_t.shape[0]
    pm = pmax[None, :].expand(B, -1).clone()
    valid = cont_t >= 0
    if valid.any():
        rows = torch.arange(B, device=device)[valid]
        cols = cont_t[valid]
        pm[rows, cols] = 0.0
    return pm


def box_clamp(p, pmin, pmax_eff):
    """Elementwise clamp into [pmin, pmax_eff] (pmax_eff already zeroed at the outage)."""
    return torch.minimum(torch.maximum(p, pmin), torch.maximum(pmax_eff, pmin))


def balance_repair(p, effective_demand_bus, pmin, pmax_eff):
    """Differentiable reserve-response repair: redistribute the power-balance residual
    over generators with available headroom. ``effective_demand_bus`` must be
    conventional PD plus fixed active shunt demand GS."""
    r = effective_demand_bus.sum(1, keepdim=True) - p.sum(1, keepdim=True)
    up = (pmax_eff - p).clamp_min(0.0)                           # headroom to increase
    dn = (p - pmin).clamp_min(0.0)                               # headroom to decrease
    head = torch.where(r >= 0, up, dn)
    tot = head.sum(1, keepdim=True).clamp_min(1e-6)
    p2 = p + r * head / tot
    return box_clamp(p2, pmin, pmax_eff)


@torch.no_grad()
def score(op, p_hat: torch.Tensor, d_bus: torch.Tensor,
          p_star: torch.Tensor | None = None, eps: float = 1e-9,
          cont_line_idx: torch.Tensor | None = None,
          cont_gen_idx: torch.Tensor | None = None) -> dict:
    """Harmonized metrics used by both DEQ and the corrected baselines.

    ``viol_MW`` is the mean across samples of each sample's worst branch,
    matching the DEQ diagnostic. Branch-average L1 violation is reported
    separately as ``viol_branch_mean_MW`` / ``J_flow_l1``.
    """
    d_eff = op.effective_demand(d_bus)
    bal = p_hat.sum(dim=1) - d_eff.sum(dim=1)
    viol_matrix = op.flow_violation_vec(p_hat, d_bus, cont_line_idx)
    worst = viol_matrix.max(dim=1).values
    m = {
        "bal_abs_MW": float(bal.abs().mean()),
        "bal_max_MW": float(bal.abs().max()),
        "bal_sat_rate": float((bal.abs() <= 1e-3).float().mean()),
        "viol_MW": float(worst.mean()),
        "viol_worst_mean_MW": float(worst.mean()),
        "viol_p95_MW": float(torch.quantile(worst, 0.95)),
        "viol_branch_mean_MW": float(viol_matrix.mean()),
        "viol_max_MW": float(viol_matrix.max()),
        "flow_sat_rate": float((worst <= 1e-3).float().mean()),
        "J_flow": float((viol_matrix ** 2).mean()),
        "J_flow_l1": float(viol_matrix.mean()),
    }
    if p_star is not None:
        cp = op.gen_cost(p_hat)
        ct = op.gen_cost(p_star)
        raw_gap = cp - ct
        abs_gap = raw_gap.abs()
        m["rmse_MW"] = float(torch.sqrt(torch.mean((p_hat - p_star) ** 2)))
        # Raw objective differences are the primary report metrics. Keep the
        # legacy ``cost_gap`` key as a backward-compatible alias for the signed
        # mean, and retain the relative gap only as a secondary diagnostic.
        m["cost_gap"] = float(raw_gap.mean())
        m["cost_gap_raw_mean"] = float(raw_gap.mean())
        m["cost_gap_raw_median"] = float(raw_gap.median())
        m["cost_gap_abs_mean"] = float(abs_gap.mean())
        m["cost_gap_abs_median"] = float(abs_gap.median())
        m["cost_gap_abs_p95"] = float(torch.quantile(abs_gap, 0.95))
        m["cost_rel_gap_abs"] = float((abs_gap / (ct.abs() + eps)).mean())
    # Mixed-contingency studies need category-specific results, not only a
    # pooled average.  Keep the original aggregate keys unchanged and append a
    # compact flat breakdown when either outage vector is supplied.
    if cont_line_idx is not None or cont_gen_idx is not None:
        n = int(p_hat.shape[0])
        line = (
            cont_line_idx.to(device=p_hat.device, dtype=torch.long).reshape(-1) >= 0
            if cont_line_idx is not None
            else torch.zeros(n, dtype=torch.bool, device=p_hat.device)
        )
        gen = (
            cont_gen_idx.to(device=p_hat.device, dtype=torch.long).reshape(-1) >= 0
            if cont_gen_idx is not None
            else torch.zeros(n, dtype=torch.bool, device=p_hat.device)
        )
        if torch.any(line & gen):
            raise ValueError("score received simultaneous line+generator outage rows")
        categories = {"clean": ~(line | gen), "line": line, "generator": gen}
        cp = op.gen_cost(p_hat) if p_star is not None else None
        ct = op.gen_cost(p_star) if p_star is not None else None
        for name, mask in categories.items():
            prefix = f"{name}_"
            count = int(mask.sum().item())
            m[prefix + "samples"] = count
            if count == 0:
                continue
            mworst = worst[mask]
            mbal = bal[mask].abs()
            m[prefix + "bal_abs_MW"] = float(mbal.mean())
            m[prefix + "bal_max_MW"] = float(mbal.max())
            m[prefix + "viol_MW"] = float(mworst.mean())
            m[prefix + "viol_p95_MW"] = float(torch.quantile(mworst, 0.95))
            m[prefix + "viol_max_MW"] = float(mworst.max())
            m[prefix + "flow_sat_rate"] = float((mworst <= 1e-3).float().mean())
            if p_star is not None and cp is not None and ct is not None:
                raw_gap = cp[mask] - ct[mask]
                abs_gap = raw_gap.abs()
                m[prefix + "rmse_MW"] = float(
                    torch.sqrt(torch.mean((p_hat[mask] - p_star[mask]) ** 2))
                )
                m[prefix + "cost_gap_raw_mean"] = float(raw_gap.mean())
                m[prefix + "cost_gap_raw_median"] = float(raw_gap.median())
                m[prefix + "cost_gap_abs_mean"] = float(abs_gap.mean())
                m[prefix + "cost_gap_abs_median"] = float(abs_gap.median())
                m[prefix + "cost_gap_abs_p95"] = float(torch.quantile(abs_gap, 0.95))
                m[prefix + "cost_rel_gap_abs"] = float(
                    (abs_gap / (ct[mask].abs() + eps)).mean()
                )
    return m


if __name__ == "__main__":
    # Scorer sanity check: feed the solver labels back through the scorer.
    # Expect ~0 balance residual, ~0 flow violation, and exactly 0 gap / RMSE.
    case = os.environ.get("DEQ_CASE", "case300")
    npz = sys.argv[1] if len(sys.argv) > 1 else "io/dc_opf_dataset_case300.npz"
    print(f"[sanity] DEQ_CASE={case}  dataset={npz}")
    op, static_np, cfg = build_operator("cpu")
    print(f"[sanity] operator built: nbus={static_np['nbus']} ngen={static_np['ngen']} nbranch={static_np['nbranch']}")
    d = load_gencont(npz)
    if d["pg"] is None:
        print("[sanity] dataset has no pg labels; cannot run scorer sanity check.")
        sys.exit(1)
    n = min(1000, d["pd_bus"].shape[0])
    d_bus = torch.from_numpy(d["pd_bus"][:n])
    p_star = torch.from_numpy(d["pg"][:n])
    cont_line = (
        torch.from_numpy(d["cont_line"][:n]) if d.get("cont_line") is not None else None
    )
    cont_gen = torch.from_numpy(d["cont"][:n]) if d.get("cont") is not None else None
    if d.get("lodf") is not None:
        op.LODF = torch.from_numpy(d["lodf"]).to(dtype=op.PTDF.dtype)
    m = score(
        op, p_star, d_bus, p_star,
        cont_line_idx=cont_line,
        cont_gen_idx=cont_gen,
    )
    print("[sanity] feeding labels p* back through scorer:")
    for k, v in m.items():
        print(f"    {k:32s} = {v:.6g}")
    ok = (abs(m["cost_rel_gap_abs"]) < 1e-6 and m["rmse_MW"] < 1e-6)
    print(f"[sanity] gap/rmse ~ 0 (self-consistency): {'PASS' if ok else 'FAIL'}")
    print(f"[sanity] balance/flow residual of solver labels: "
          f"bal={m['bal_abs_MW']:.4g} MW, viol={m['viol_MW']:.4g} MW "
          f"(small => trainer physics matches the dataset's solver)")
