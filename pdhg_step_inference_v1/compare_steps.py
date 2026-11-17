"""Fixed-depth and fixed-time learned-versus-initial-PDHG comparison.

The trained checkpoint remains immutable. Both arms use the identical operator,
test split, starting state, numerical precision, and scoring contract.  The
only changed object is ``LearnableSteps``: checkpoint-learned versus the
original safe scalar initialization (rho=0.9, s=0).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import paired_budget_rows


def parse_ints(value: str) -> list[int]:
    out = sorted({int(x) for x in value.split(",") if x.strip()})
    if not out or min(out) < 1:
        raise ValueError("depths must be positive")
    return out


def parse_floats(value: str) -> list[float]:
    out = sorted({float(x) for x in value.split(",") if x.strip()})
    if not out or min(out) <= 0:
        raise ValueError("budgets must be positive")
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def validate_generic_source(base, data_path: Path, audit_path: Path, checkpoint_path: Path,
                            summary_path: Path, case_name: str) -> tuple[dict, dict, dict]:
    """Validate a phase-aware production triplet without a hard-coded hash list.

    The audit binds its recorded output hash to the submitted data. All hashes
    are retained in the result provenance, including the checkpoint and source
    summary. This allows case9241pegase to participate even though the older
    case300 fixed-K script did not contain a pinned hash entry for it.
    """
    for path in (data_path, audit_path, checkpoint_path, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text())
    summary = json.loads(summary_path.read_text())
    hashes = {
        "dataset_sha256": base.sha256_file(data_path),
        "audit_sha256": base.sha256_file(audit_path),
        "checkpoint_sha256": base.sha256_file(checkpoint_path),
        "source_summary_sha256": base.sha256_file(summary_path),
    }
    if audit.get("case") != case_name:
        raise ValueError(f"audit case mismatch: {audit.get('case')} != {case_name}")
    if audit.get("protocol") != "fair4net_l1_v2_phase_aware":
        raise ValueError("the phase-aware data audit is required")
    if audit.get("output_sha256") != hashes["dataset_sha256"]:
        raise ValueError("dataset hash differs from its audit")
    if summary.get("case_name") != case_name:
        raise ValueError("source summary case mismatch")
    cfg = dict(summary.get("config") or {})
    if cfg.get("step_mode") != "scalar" or float(cfg.get("w_flow_l1", 0.0)) <= 0.0:
        raise ValueError("source is not the scalar L1-DEQ configuration")
    return audit, summary, hashes


def steps_record(steps, dtype: torch.dtype, device: torch.device) -> dict:
    tau, sigma = steps.tau_sigma(dtype, device)
    return {
        "rho": float(steps.rho().detach().cpu()),
        "s": float(steps.s().detach().cpu()),
        "tau": float(tau.detach().cpu()),
        "sigma": float(sigma.detach().cpu()),
        "tau_sigma_nK2": float(tau.detach().cpu() * sigma.detach().cpu() * steps.nK * steps.nK),
    }


def make_initial_step_operator(learned_op, trainer, checkpoint: dict, fixed_rho: float, fixed_s: float, device):
    """Clone immutable physics, then replace only the scalar step parameters."""
    fixed = copy.deepcopy(learned_op)
    source_steps = learned_op.steps
    steps = trainer.LearnableSteps(
        nK=float(checkpoint["nK"]),
        s_max=float(source_steps.s_max),
        rho_init=fixed_rho,
        s_init=fixed_s,
    ).to(device)
    for param in steps.parameters():
        param.requires_grad_(False)
    fixed.steps = steps
    fixed.eval()
    return fixed


def quality(base, harness, op, data, indices: np.ndarray, depth: int, batch_size: int, device) -> dict:
    pred, demand, target, residuals = [], [], [], []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        d = torch.as_tensor(data["pd_bus"][ids], dtype=torch.float32, device=device)
        p_star = torch.as_tensor(data["pg"][ids], dtype=torch.float32, device=device)
        cont = None if data["cont"] is None else torch.as_tensor(data["cont"][ids], dtype=torch.long, device=device)
        z = base.rollout_fixed_k(op, d, cont, depth)
        nxt = op(z, d, cont)
        residuals.append((nxt - z).norm(dim=1) / (nxt.norm(dim=1) + 1e-12))
        pred.append(z[:, :op.ngen])
        demand.append(d)
        target.append(p_star)
    p = torch.cat(pred)
    d = torch.cat(demand)
    p_star = torch.cat(target)
    metrics = harness.score(op, p, d, p_star)
    metrics["cost_gap_abs_percent"] = 100.0 * metrics["cost_rel_gap_abs"]
    metrics["fixed_point_residual_mean"] = float(torch.cat(residuals).mean())
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True, type=Path, help="fair4net_l1_study directory")
    ap.add_argument("--case-name", default="case300", choices=("case300", "case1354pegase", "case2869pegase", "case9241pegase"))
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--source-stem", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depths", default="1,2,4,8,16,32,64,128,200")
    ap.add_argument("--budgets-ms", default="1,2,5,10,20,50,100")
    ap.add_argument("--timing-batch-size", type=int, default=16)
    ap.add_argument("--quality-batch-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--fixed-rho", type=float, default=0.9)
    ap.add_argument("--fixed-s", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.timing_batch_size < 1 or args.quality_batch_size < 1 or args.repeats < 1:
        raise ValueError("invalid batch or repeat count")
    if not (0.0 < args.fixed_rho < 0.99):
        raise ValueError("fixed rho must lie in (0, 0.99)")
    # The canonical trainer selects its network at import time.
    os.environ["DEQ_CASE"] = args.case_name
    sys.path.insert(0, str(args.study_dir.resolve()))
    import case300_fixed_k_inference_budget as base
    import deq_trainer_fair4net as trainer
    import baseline_harness_fair4net as harness

    if trainer.CASE_NAME != args.case_name:
        raise ValueError(f"trainer selected {trainer.CASE_NAME}, expected {args.case_name}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(str(args.source_stem) + ".pt")
    summary_path = Path(str(args.source_stem) + "_summary.json")
    audit, summary, hashes = validate_generic_source(
        base, args.data, args.audit, ckpt_path, summary_path, args.case_name
    )
    learned, data, checkpoint, flow_provenance = base.build_operator_and_data(
        args.data, audit, summary, ckpt_path, args.case_name, device, False
    )
    fixed = make_initial_step_operator(learned, trainer, checkpoint, args.fixed_rho, args.fixed_s, device)
    arms = {"learned_steps": learned, "initial_safe_steps": fixed}
    test = np.flatnonzero(np.asarray(data["split"], dtype=np.int64) == 2)
    if args.max_test_samples:
        test = test[:args.max_test_samples]
    if len(test) < args.timing_batch_size:
        raise ValueError("test subset smaller than timing batch")
    depths = parse_ints(args.depths)
    budgets = parse_floats(args.budgets_ms)
    rows: list[dict] = []
    timing: dict[str, dict[int, float]] = {name: {} for name in arms}
    for name, op in arms.items():
        for depth in depths:
            ids = test[:args.timing_batch_size]
            d = torch.as_tensor(data["pd_bus"][ids], dtype=torch.float32, device=device)
            cont = None if data["cont"] is None else torch.as_tensor(data["cont"][ids], dtype=torch.long, device=device)
            times = base.timed_batch(op, d, cont, depth, args.warmup, args.repeats, device)
            median = float(np.median(times))
            timing[name][depth] = median
            metrics = quality(base, harness, op, data, test, depth, args.quality_batch_size, device)
            rows.append({"arm": name, "depth": depth, "latency_median_ms": median, "latency_p95_ms": float(np.percentile(times, 95)), "test_samples": int(len(test)), **metrics})
            print(f"[pdhg-steps] {name} K={depth} median_ms={median:.4g} rmse={metrics['rmse_MW']:.5g} viol={metrics['viol_worst_mean_MW']:.5g}")
    by_arm_depth = {(r["arm"], r["depth"]): r for r in rows}
    budgets_out: list[dict] = []
    for pair in paired_budget_rows(timing["learned_steps"], timing["initial_safe_steps"], budgets):
        output = dict(pair)
        for arm, key in (("learned_steps", "learned"), ("initial_safe_steps", "fixed")):
            depth = pair[f"{key}_max_k"]
            result = by_arm_depth.get((arm, depth)) if depth is not None else None
            for metric in ("rmse_MW", "cost_gap_abs_mean", "cost_gap_abs_percent", "bal_abs_MW", "viol_worst_mean_MW", "viol_p95_MW", "viol_max_MW", "flow_sat_rate", "fixed_point_residual_mean"):
                output[f"{key}_{metric}"] = None if result is None else result[metric]
        budgets_out.append(output)
    provenance = {
        "status": "PASS",
        "experiment": f"{args.case_name} solver-only learned versus initial-safe PDHG steps",
        "case_name": args.case_name,
        "fairness": "identical checkpoint, operator, zero initial state, test rows, precision, batch sizes, and scorer; only the scalar PDHG step pair differs",
        "fixed_arm_definition": "original safe scalar initialization, never trained: rho=0.9, s=0",
        "learned_arm_definition": "checkpoint scalar steps_state",
        "learned_steps": steps_record(learned.steps, torch.float32, device),
        "initial_safe_steps": steps_record(fixed.steps, torch.float32, device),
        "test_samples": int(len(test)),
        "hashes": hashes,
        "flow_offset_provenance": flow_provenance,
        "smoke": bool(args.smoke),
    }
    write_csv(args.out_dir / "paired_depth_metrics.csv", rows)
    write_csv(args.out_dir / "fixed_time_metrics.csv", budgets_out)
    write_json(args.out_dir / "result.json", {"provenance": provenance, "by_depth": rows, "fixed_time": budgets_out})
    print("[pdhg-steps] PASS", args.out_dir)


if __name__ == "__main__":
    main()
