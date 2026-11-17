#!/usr/bin/env python3
"""Fail fast when a DEQ smoke or final result is missing/non-finite."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEQ_CODE_VERSION = "fair4net-l1-v2-phase-aware-resumable"
PROTOCOL = "fair4net_l1_v2_phase_aware"
FLOW_MODEL_CONTRACT = "f=PTDF@(Pg-PD-GS)+phase_shift_offset"
PARTIAL_RESIDUAL_LIMIT = 1e-4
FINAL_RESIDUAL_LIMIT = 5e-5


def finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--partial-epoch", type=int)
    args = ap.parse_args()

    summary_path = args.out_dir / f"{args.stem}_summary.json"
    history_path = args.out_dir / f"{args.stem}_latest.json"
    if not summary_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Missing DEQ summary/history under {args.out_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if summary.get("case_name") != args.case:
        raise ValueError(f"Summary case mismatch: {summary.get('case_name')} != {args.case}")
    if summary.get("code_version") != DEQ_CODE_VERSION:
        raise ValueError(
            "Stale DEQ code version: "
            f"{summary.get('code_version')!r} != {DEQ_CODE_VERSION!r}"
        )
    physics = summary.get("physics", {})
    if physics.get("code_version") != DEQ_CODE_VERSION:
        raise ValueError(
            "Stale DEQ physics version: "
            f"{physics.get('code_version')!r} != {DEQ_CODE_VERSION!r}"
        )
    if physics.get("flow_model_contract") != FLOW_MODEL_CONTRACT:
        raise ValueError(
            "DEQ flow-model contract mismatch: "
            f"{physics.get('flow_model_contract')!r} != {FLOW_MODEL_CONTRACT!r}"
        )
    flow_offset_sha = physics.get("flow_offset_sha256")
    if not isinstance(flow_offset_sha, str) or len(flow_offset_sha) != 64:
        raise ValueError(f"Invalid DEQ flow-offset digest: {flow_offset_sha!r}")

    audit_path = args.out_dir.parent / "dataset_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Stale dataset audit protocol: {audit.get('protocol')!r} != {PROTOCOL!r}"
        )
    if audit.get("flow_model_contract") != FLOW_MODEL_CONTRACT:
        raise ValueError(
            "Dataset-audit flow-model contract mismatch: "
            f"{audit.get('flow_model_contract')!r} != {FLOW_MODEL_CONTRACT!r}"
        )
    if audit.get("flow_offset_sha256") != flow_offset_sha:
        raise ValueError(
            "DEQ/dataset flow-offset digest mismatch: "
            f"{flow_offset_sha!r} != {audit.get('flow_offset_sha256')!r}"
        )
    if not history:
        raise ValueError("DEQ history is empty.")

    last_epoch = int(history[-1]["epoch"])
    expected = args.partial_epoch if args.partial_epoch is not None else 30
    if last_epoch != expected:
        raise ValueError(f"Expected last epoch {expected}, got {last_epoch}.")

    cfg = summary.get("config", {})
    required_cfg = {
        "step_mode": "scalar",
        "w_cost": 1.0,
        "w_bal": 1e5,
        "w_flow": 1e8,
        "w_flow_l1": 1e4,
    }
    for key, wanted in required_cfg.items():
        got = cfg.get(key)
        if got != wanted:
            raise ValueError(f"Unexpected {key}: got {got!r}, expected {wanted!r}")

    val = history[-1].get("val", {})
    for key in ("L_total", "fwd_res", "alg_res", "bal_abs_MW", "viol_MW"):
        finite(val.get(key), f"last.val.{key}")
    if args.partial_epoch is not None:
        # The one-epoch smoke validates stable finite convergence, not the
        # tighter best-checkpoint accuracy required of the finished run.
        if finite(val["fwd_res"], "last.val.fwd_res") > PARTIAL_RESIDUAL_LIMIT:
            raise ValueError(f"Forward residual gate failed: {val['fwd_res']}")
        if finite(val["alg_res"], "last.val.alg_res") > PARTIAL_RESIDUAL_LIMIT:
            raise ValueError(f"Algebraic residual gate failed: {val['alg_res']}")

    if args.partial_epoch is None:
        if not 1 <= int(summary.get("best_epoch", 0)) <= 30:
            raise ValueError(f"Invalid best_epoch: {summary.get('best_epoch')}")
        test = summary.get("test", {})
        for key in (
            "L_total",
            "fwd_res",
            "alg_res",
            "J_flow_l1",
            "bal_abs_MW",
            "bal_max_MW",
            "viol_MW",
            "viol_p95_MW",
            "viol_max_MW",
            "rmse_MW_diag",
            "cost_rel_gap_abs",
            "flow_sat_rate",
        ):
            finite(test.get(key), f"test.{key}")
        if (
            float(test["fwd_res"]) > FINAL_RESIDUAL_LIMIT
            or float(test["alg_res"]) > FINAL_RESIDUAL_LIMIT
        ):
            raise ValueError(
                "Best-checkpoint residual gate failed: "
                f"fwd={test['fwd_res']} alg={test['alg_res']}"
            )
        resources = summary.get("resources", {})
        if (
            int(resources.get("train_samples", -1)),
            int(resources.get("val_samples", -1)),
            int(resources.get("test_samples", -1)),
        ) != (7000, 1000, 2000):
            raise ValueError(f"Unexpected split counts in resources: {resources}")
        ev = args.out_dir / f"{args.stem}_eval_vs_gt_summary.json"
        if not ev.is_file():
            raise FileNotFoundError(ev)
        ev_summary = json.loads(ev.read_text(encoding="utf-8"))
        if int(ev_summary.get("N", -1)) != 2000:
            raise ValueError(f"Expected 2000 per-sample test rows, got {ev_summary.get('N')}")

    print(
        f"[validate] PASS case={args.case} last_epoch={last_epoch} "
        f"best_epoch={summary.get('best_epoch')}"
    )


if __name__ == "__main__":
    main()
