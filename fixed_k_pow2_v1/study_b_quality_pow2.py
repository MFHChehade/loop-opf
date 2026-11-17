#!/usr/bin/env python3
"""Quality-only power-of-two mixed-contingency fixed-K rollout.

Reuses the validated campaign loader and the existing single-rollout quality
routine. It intentionally omits the expensive repeated latency sweep; job
901788 remains the latency source for the manuscript.
"""
from __future__ import annotations

import csv
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


NINE_K_SIDECAR = Path("io/case9241_clean_recovery_v5/flow_offset_audited.npy")
NINE_K_SIDECAR_SHA256 = "3fcd6f8f7405741e65274df802de1f7fd83f19e84a95fe34f0f30d663549b401"


def check_reference(rows: list[dict], reference: Path | None) -> None:
    if reference is None:
        return
    if not reference.is_file():
        raise FileNotFoundError(reference)
    with reference.open(newline="") as stream:
        old = {(r["category"], int(r["k"])): r for r in csv.DictReader(stream)}
    for row in rows:
        key = (row["category"], int(row["k"]))
        if key not in old:
            continue
        for metric in ("rmse_MW", "bal_abs_MW", "viol_worst_mean_MW"):
            expected = float(old[key][metric])
            actual = float(row[metric])
            limit = max(1e-4, 1e-4 * abs(expected))
            if not np.isfinite(actual) or abs(actual - expected) > limit:
                raise RuntimeError(
                    f"mixed parity failure {key} {metric}: new={actual} old={expected} limit={limit}"
                )
    print(f"[study-b] reference parity PASS {reference}", flush=True)


def main() -> None:
    # Both modules are in fair4net_l1_study on Vista. DEQ_CASE must be exported
    # by the SLURM wrapper before this process starts.
    sys.path.insert(0, str(Path("fair4net_l1_study").resolve()))
    import mixed_fixed_k_ablation as B

    parser = B.build_parser()
    parser.add_argument("--reference-quality", type=Path)
    args = parser.parse_args()
    if os.environ.get("DEQ_CASE") != args.case:
        raise ValueError("DEQ_CASE must select the requested network")
    if args.warmup != 1 or args.repeats != 3:
        raise ValueError("keep the campaign's existing compatibility settings")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output: {args.out_dir}")

    import baseline_harness_fair4net as H
    import deq_trainer_fair4net as tr
    import mixed_deq_inference as M

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Vista GPU required")

    audit, meta, summary, data, hashes = M.validate_sources(args, tr, H)
    build_args = args
    if args.case == "case9241pegase":
        # The shared builder enforces a bytewise flow hash before this wrapper
        # can apply the pinned-sidecar numerical gate. Defer only its early
        # hash checks; the full-data gate above and exact LODF / numerical flow
        # gates below remain mandatory, including for production runs.
        if M.sha256_file(Path(M.__file__)) != "8369f14101ce74104a41570c51f4f8daaaf7727c6160bb3693a69e46e7e4ad28":
            raise ValueError("shared model builder source changed")
        build_args = copy.copy(args)
        build_args.smoke = True
    cfg, op, _deq, checkpoint, physics = M.build_model(build_args, tr, H, data, summary, device)
    flow_ok = (physics["runtime_flow_offset_sha256"] == physics["checkpoint_flow_offset_sha256"]
               or physics["accepted_established_thread_equivalent_pair"])
    lodf_ok = physics["runtime_lodf_sha256"] == physics["checkpoint_lodf_sha256"]
    if not flow_ok and args.case == "case9241pegase":
        if not NINE_K_SIDECAR.is_file() or M.sha256_file(NINE_K_SIDECAR) != NINE_K_SIDECAR_SHA256:
            raise ValueError("9k audited flow-offset sidecar is missing or changed")
        audited = np.load(NINE_K_SIDECAR)
        runtime = op.flow_offset.detach().cpu().numpy()
        if audited.shape != runtime.shape:
            raise ValueError("9k audited flow-offset shape differs from runtime")
        error_mw = float(np.max(np.abs(runtime - audited)))
        if not np.isfinite(error_mw) or error_mw > 1e-5:
            raise ValueError(f"9k audited flow-offset mismatch: {error_mw} MW")
        physics["audited_sidecar_sha256"] = NINE_K_SIDECAR_SHA256
        physics["audited_sidecar_max_abs_error_MW"] = error_mw
        physics["audited_sidecar_tolerance_MW"] = 1e-5
        flow_ok = True
    if not flow_ok or not lodf_ok:
        raise ValueError(f"strict flow-offset or LODF provenance mismatch: {json.dumps(physics, sort_keys=True)}")
    test_groups = M.category_indices(data, 2)
    counts = {key: int(value.size) for key, value in test_groups.items()}
    if not args.smoke and counts["pooled"] != args.expected_test:
        raise ValueError(f"expected {args.expected_test} pooled test rows, got {counts['pooled']}")
    if not args.k_values or any(k < 1 or k & (k - 1) for k in args.k_values):
        raise ValueError("all K values must be powers of two")
    rows = B.evaluate_quality(args, M, H, op, data, test_groups, device)
    if args.smoke:
        expected_rows = len(args.k_values) * len(B.CATEGORIES)
    else:
        expected_rows = len(args.k_values) * len(B.CATEGORIES)
        for k in args.k_values:
            category_counts = {r["category"]: r["samples"] for r in rows if r["k"] == k}
            if category_counts.get("pooled") != args.expected_test:
                raise ValueError(f"K={k} has an incomplete pooled test set")
            if sum(category_counts.get(c, 0) for c in ("clean", "line", "generator")) != args.expected_test:
                raise ValueError(f"K={k} category counts do not partition the test set")
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} quality rows, got {len(rows)}")
    check_reference(rows, args.reference_quality)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    B.write_csv(args.out_dir / "quality_by_k.csv", rows)
    B.write_json(args.out_dir / "fixed_k_manifest.json", {
        "status": "PASS", "study": "mixed_n1_pow2_quality_only",
        "case": args.case, "K": args.k_values,
        "initial_state": "z0=0", "anderson": False,
        "residual_stopping": False, "dtype": "float32",
        "quality_batch_size": args.quality_batch_size or M.QUALITY_BATCH[args.case],
        "quality_test_rows": counts, "source_hashes": hashes,
        "physics_provenance": physics,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_nK": float(checkpoint["nK"]),
        "latency_source": "existing job 901788; no latency timing in this run",
        "reference_quality": str(args.reference_quality) if args.reference_quality else None,
    })
    print(f"[study-b] PASS case={args.case} K={args.k_values} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
