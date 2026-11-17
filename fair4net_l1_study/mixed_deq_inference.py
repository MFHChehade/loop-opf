#!/usr/bin/env python3
"""Inference/runtime study for completed mixed N-1 scalar L1-DEQ checkpoints.

The primary timing protocol intentionally preserves the prior line-only study:
one untimed warm-up and three synchronized repetitions.  It extends that
protocol to pooled, clean, unseen-line, and unseen-generator test rows and to
batch sizes 1, 8, 32, plus a validation-discovered safe maximum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mixed_inference_utils import CATEGORIES, aggregate_timing, parse_positive_ints


FLOW_TOL_MW = 1e-3
EXPECTED_PROTOCOL = "fair4net_mixed_n1_contingency_v1"
CASES = ("case300", "case1354pegase", "case2869pegase", "case9241pegase")
QUALITY_BATCH = {"case300": 16, "case1354pegase": 8, "case2869pegase": 8, "case9241pegase": 4}
THREAD_EQUIVALENT_FLOW_HASH_PAIRS = {
    "case1354pegase": (
        "2732941bb151044da609805c6615e41259d258c3a42c1374c6b5add6a0f3eb8e",
        "a6cc70f48b157d46e08e923994ba60d2c93c53fea7b373c2a47ad7f0b6632fb2",
    ),
    "case2869pegase": (
        "c9af3e8fde5b190991454dcef79f3ef371509da8b33c8fa7bbd55bd1e4adfc66",
        "48c88fc0d88ac335576470e381a2b3ea76b11f432276ae1a84fb6bee15b4691b",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def category_indices(data: dict[str, Any], split_value: int) -> dict[str, np.ndarray]:
    split = np.asarray(data["split"], dtype=np.int64)
    line = np.asarray(data["cont_line"], dtype=np.int64)
    gen = np.asarray(data["cont"], dtype=np.int64)
    if np.any((line >= 0) & (gen >= 0)):
        raise ValueError("dataset contains simultaneous line and generator outages")
    selected = split == int(split_value)
    return {
        "pooled": np.flatnonzero(selected),
        "clean": np.flatnonzero(selected & (line < 0) & (gen < 0)),
        "line": np.flatnonzero(selected & (line >= 0) & (gen < 0)),
        "generator": np.flatnonzero(selected & (line < 0) & (gen >= 0)),
    }


def validate_sources(args, tr, H) -> tuple[dict, dict, dict, dict]:
    paths = [args.data, args.audit, args.meta, args.checkpoint, args.summary]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    audit = read_json(args.audit)
    meta = read_json(args.meta)
    summary = read_json(args.summary)
    if audit.get("status") != "PASS" or audit.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("dataset audit is not a passing mixed N-1 production audit")
    if audit.get("case") != args.case or meta.get("case_name") != args.case:
        raise ValueError("dataset audit/metadata case mismatch")
    if meta.get("scenario_contract") != "exactly one of clean, single-line outage, single-generator outage; never simultaneous":
        raise ValueError("mixed scenario contract mismatch")
    if meta.get("demand_convention") != "pd_bus is conventional load; effective demand=pd_bus+gs_bus":
        raise ValueError("effective-demand contract mismatch")
    if meta.get("flow_model_contract") != tr.FLOW_MODEL_CONTRACT:
        raise ValueError("flow model contract mismatch")
    if summary.get("case_name") != args.case:
        raise ValueError("checkpoint summary case mismatch")
    cfg = dict(summary.get("config") or {})
    expected = {
        "step_mode": "scalar", "picard_iters": 50, "anderson_m": 4,
        "anderson_lam": 1e-3, "anderson_max_iter": 600, "anderson_tol": 1e-10,
        "w_flow": 1e8, "w_flow_l1": 1e4,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"solver/checkpoint contract mismatch: {key}={cfg.get(key)!r}")

    data = H.load_gencont(str(args.data))
    if data["pg"] is None or data["split"] is None or data["lodf"] is None:
        raise ValueError("mixed inference requires labels, fixed split, and audited LODF")
    n = int(data["pd_bus"].shape[0])
    if not args.smoke and n != 10000:
        raise ValueError(f"production dataset must contain 10000 rows, found {n}")

    hashes = {
        "dataset_sha256": sha256_file(args.data),
        "audit_sha256": sha256_file(args.audit),
        "metadata_sha256": sha256_file(args.meta),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "summary_sha256": sha256_file(args.summary),
    }
    return audit, meta, summary, data, hashes


def build_model(args, tr, H, data, summary, device):
    net = tr.build_base_net()
    ppc = tr.build_ppc_from_net(net)
    static_np = tr.extract_static_mats(ppc)
    static_np["lodf"] = np.asarray(data["lodf"], dtype=np.float32)
    H.validate_contract(data, static_np, require_metadata=True)
    physics = dict(summary.get("physics") or {})
    runtime_flow = tr.sha256_array(static_np["flow_offset"])
    source_flow = physics.get("flow_offset_sha256")
    accepted_pair = False
    if runtime_flow != source_flow:
        if (runtime_flow, source_flow) == THREAD_EQUIVALENT_FLOW_HASH_PAIRS.get(args.case):
            accepted_pair = True
        elif not args.smoke:
            raise ValueError(f"flow-offset hash mismatch: runtime={runtime_flow} summary={source_flow}")
    runtime_lodf = tr.sha256_array(static_np["lodf"])
    if runtime_lodf != physics.get("lodf_sha256") and not args.smoke:
        raise ValueError("audited LODF hash differs from checkpoint summary")

    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("case_name") != args.case or "steps_state" not in checkpoint or "nK" not in checkpoint:
        raise ValueError("checkpoint identity or learned-step state is invalid")
    old_cfg = dict(checkpoint.get("cfg") or {})
    if old_cfg.get("step_mode") != "scalar":
        raise ValueError("only the publication scalar checkpoint is allowed")
    source_cfg = dict(summary["config"])
    cfg = tr.TrainCfg(
        device=str(device), dtype=torch.float32,
        picard_iters=int(source_cfg["picard_iters"]),
        anderson_m=int(source_cfg["anderson_m"]),
        anderson_lam=float(source_cfg["anderson_lam"]),
        anderson_max_iter=int(source_cfg["anderson_max_iter"]),
        anderson_tol=float(source_cfg["anderson_tol"]),
        anderson_beta=float(source_cfg.get("anderson_beta", 1.0)),
        b_anderson_m=int(source_cfg.get("b_anderson_m", 10)),
        b_anderson_lam=float(source_cfg.get("b_anderson_lam", 1e-4)),
        b_anderson_max_iter=int(source_cfg.get("b_anderson_max_iter", 100)),
        b_anderson_tol=float(source_cfg.get("b_anderson_tol", 1e-6)),
    )
    static = tr.to_torch_static(static_np, cfg)
    steps = tr.LearnableSteps(
        nK=float(checkpoint["nK"]),
        s_max=float(old_cfg.get("s_max", cfg.s_max)), rho_init=0.9, s_init=0.0,
    ).to(device)
    steps.load_state_dict(checkpoint["steps_state"], strict=True)
    op = tr.PDHGOperatorLearnable(static, steps).to(device).eval()
    dim = int(static["ngen"]) + 1 + 2 * int(static["nbranch"])
    deq = tr.DEQFixedPoint(
        f=op, dim=dim, picard_iters=cfg.picard_iters,
        fwd_kwargs=dict(m=cfg.anderson_m, lam=cfg.anderson_lam,
                        max_iter=cfg.anderson_max_iter, tol=cfg.anderson_tol,
                        beta=cfg.anderson_beta),
        bwd_kwargs=dict(m=cfg.b_anderson_m, lam=cfg.b_anderson_lam,
                        max_iter=cfg.b_anderson_max_iter, tol=cfg.b_anderson_tol,
                        beta=cfg.b_anderson_beta),
    ).to(device).eval()
    provenance = {
        "runtime_flow_offset_sha256": runtime_flow,
        "checkpoint_flow_offset_sha256": source_flow,
        "accepted_established_thread_equivalent_pair": accepted_pair,
        "runtime_lodf_sha256": runtime_lodf,
        "checkpoint_lodf_sha256": physics.get("lodf_sha256"),
    }
    return cfg, op, deq, checkpoint, provenance


def batch_numpy(data, indices):
    return (
        np.ascontiguousarray(data["pd_bus"][indices], dtype=np.float32),
        np.ascontiguousarray(data["cont"][indices], dtype=np.int64),
        np.ascontiguousarray(data["cont_line"][indices], dtype=np.int64),
    )


def to_device_batch(arrays, device):
    d_np, gen_np, line_np = arrays
    return (
        torch.as_tensor(d_np, dtype=torch.float32, device=device),
        torch.as_tensor(gen_np, dtype=torch.long, device=device),
        torch.as_tensor(line_np, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def run_deq(deq, d, gen, line):
    return deq(d, gen, line)


@torch.inference_mode()
def diagnostics_for_output(op, deq, z, d, gen, line):
    z_next = op(z, d, gen, line)
    alg = (z_next - z).norm(dim=1) / (1e-12 + z_next.norm(dim=1))
    fwd = float(deq.forward_res[-1]) if deq.forward_res else float("nan")
    calls = int(deq.picard_iters + 3 + len(deq.forward_res))
    return calls, fwd, float(alg.mean().item()), int(deq.forward_fp64_fallback_rows)


def discover_batches(deq, data, val_indices, requested, candidates, device):
    rows = []
    for batch_size in sorted(set(requested + candidates)):
        if batch_size > val_indices.size:
            rows.append({"batch_size": batch_size, "status": "insufficient_validation_rows", "peak_cuda_mem_alloc_bytes": 0})
            continue
        arrays = batch_numpy(data, val_indices[:batch_size])
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            d, gen, line = to_device_batch(arrays, device)
            run_deq(deq, d, gen, line)
            sync(device)
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            rows.append({"batch_size": batch_size, "status": "PASS", "peak_cuda_mem_alloc_bytes": peak})
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append({"batch_size": batch_size, "status": "CUDA_OOM", "peak_cuda_mem_alloc_bytes": 0})
    passed = [int(r["batch_size"]) for r in rows if r["status"] == "PASS"]
    if not passed:
        raise RuntimeError("no timing batch size passed validation capacity discovery")
    safe_max = max(passed)
    selected = sorted({b for b in requested if b in passed} | {safe_max})
    return rows, selected


def measure_timings(args, deq, op, data, groups, batch_sizes, device):
    rows = []
    for category in CATEGORIES:
        available = groups[category]
        for batch_size in batch_sizes:
            if batch_size > available.size:
                continue
            arrays = batch_numpy(data, available[:batch_size])
            d, gen, line = to_device_batch(arrays, device)
            for _ in range(args.warmup):
                run_deq(deq, d, gen, line)
            sync(device)
            for repeat in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    z = run_deq(deq, d, gen, line)
                    end.record()
                    end.synchronize()
                    latency_ms = float(start.elapsed_time(end))
                    peak = int(torch.cuda.max_memory_allocated(device))
                else:
                    start_cpu = time.perf_counter_ns()
                    z = run_deq(deq, d, gen, line)
                    latency_ms = (time.perf_counter_ns() - start_cpu) / 1e6
                    peak = 0
                calls, fwd, alg, fallback = diagnostics_for_output(op, deq, z, d, gen, line)
                rows.append({
                    "case": args.case, "method": "deq", "category": category,
                    "batch_size": batch_size, "timing_kind": "model_only",
                    "repeat": repeat, "latency_ms": latency_ms,
                    "operator_calls": calls, "forward_residual": fwd,
                    "algebraic_residual": alg, "fp64_fallback_rows": fallback,
                    "peak_cuda_mem_alloc_bytes": peak,
                })

            # End-to-end includes CPU tensor preparation, H2D copies, DEQ, and
            # the generator output D2H copy. It excludes physics scoring and I/O.
            for _ in range(args.warmup):
                d2, gen2, line2 = to_device_batch(arrays, device)
                z2 = run_deq(deq, d2, gen2, line2)
                _ = z2[:, : int(op.ngen)].cpu()
            sync(device)
            for repeat in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                start_wall = time.perf_counter_ns()
                d2, gen2, line2 = to_device_batch(arrays, device)
                z2 = run_deq(deq, d2, gen2, line2)
                _ = z2[:, : int(op.ngen)].cpu()
                sync(device)
                latency_ms = (time.perf_counter_ns() - start_wall) / 1e6
                peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                calls, fwd, alg, fallback = diagnostics_for_output(op, deq, z2, d2, gen2, line2)
                rows.append({
                    "case": args.case, "method": "deq", "category": category,
                    "batch_size": batch_size, "timing_kind": "end_to_end",
                    "repeat": repeat, "latency_ms": latency_ms,
                    "operator_calls": calls, "forward_residual": fwd,
                    "algebraic_residual": alg, "fp64_fallback_rows": fallback,
                    "peak_cuda_mem_alloc_bytes": peak,
                })
    return rows


@torch.inference_mode()
def evaluate_accuracy(args, H, op, deq, data, groups, device):
    max_samples = int(args.max_test_samples)
    pooled = groups["pooled"]
    if max_samples > 0 and max_samples < pooled.size:
        # Stratified smoke subset, never used for production claims.
        selected = []
        each = max(1, max_samples // 3)
        for category in ("clean", "line", "generator"):
            selected.extend(groups[category][:each].tolist())
        pooled = np.asarray(selected[:max_samples], dtype=np.int64)
    predictions = []
    demands = []
    targets = []
    gen_all = []
    line_all = []
    sample_alg = []
    batch_fwd = []
    batch_calls = []
    start = time.perf_counter()
    batch_size = int(args.quality_batch_size or QUALITY_BATCH[args.case])
    for offset in range(0, pooled.size, batch_size):
        idx = pooled[offset: offset + batch_size]
        arrays = batch_numpy(data, idx)
        d, gen, line = to_device_batch(arrays, device)
        target = torch.as_tensor(data["pg"][idx], dtype=torch.float32, device=device)
        z = run_deq(deq, d, gen, line)
        z_next = op(z, d, gen, line)
        alg = (z_next - z).norm(dim=1) / (1e-12 + z_next.norm(dim=1))
        predictions.append(z[:, : int(op.ngen)])
        demands.append(d)
        targets.append(target)
        gen_all.append(gen)
        line_all.append(line)
        sample_alg.append(alg)
        batch_fwd.append(float(deq.forward_res[-1]) if deq.forward_res else float("nan"))
        batch_calls.append(int(deq.picard_iters + 3 + len(deq.forward_res)))
    sync(device)
    elapsed = time.perf_counter() - start
    p = torch.cat(predictions)
    d = torch.cat(demands)
    target = torch.cat(targets)
    gen = torch.cat(gen_all)
    line = torch.cat(line_all)
    alg = torch.cat(sample_alg)
    category_masks = {
        "pooled": torch.ones(p.shape[0], dtype=torch.bool, device=device),
        "clean": (line < 0) & (gen < 0),
        "line": (line >= 0) & (gen < 0),
        "generator": (line < 0) & (gen >= 0),
    }
    rows = []
    for category in CATEGORIES:
        mask = category_masks[category]
        metrics = H.score(op, p[mask], d[mask], target[mask],
                          cont_line_idx=line[mask], cont_gen_idx=gen[mask])
        rows.append({
            "case": args.case, "method": "deq", "category": category,
            "samples": int(mask.sum().item()),
            "rmse_MW": metrics["rmse_MW"],
            "cost_gap_abs_percent": 100.0 * metrics["cost_rel_gap_abs"],
            "viol_worst_mean_MW": metrics["viol_worst_mean_MW"],
            "viol_p95_MW": metrics["viol_p95_MW"],
            "viol_max_MW": metrics["viol_max_MW"],
            "flow_sat_rate": metrics["flow_sat_rate"],
            "bal_abs_MW": metrics["bal_abs_MW"],
            "bal_max_MW": metrics["bal_max_MW"],
            "bal_sat_rate": metrics["bal_sat_rate"],
            "algebraic_residual_mean": float(alg[mask].mean().item()),
            "forward_residual_pooled_batch_mean": statistics.mean(batch_fwd),
            "operator_calls_batch_mean": statistics.mean(batch_calls),
            "quality_pass_elapsed_sec": elapsed if category == "pooled" else None,
        })
    return rows


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sizes", type=parse_positive_ints, default=parse_positive_ints("1,8,32"))
    ap.add_argument("--safe-batch-candidates", type=parse_positive_ints, default=parse_positive_ints("64,128,256"))
    ap.add_argument("--quality-batch-size", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--expected-test", type=int, default=2000)
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--contract-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    if os.environ.get("DEQ_CASE") != args.case:
        raise ValueError("set DEQ_CASE to the requested --case before Python starts")
    if args.warmup != 1 or args.repeats != 3:
        raise ValueError("primary compatibility protocol requires warmup=1 and repeats=3")
    # Import after DEQ_CASE is verified because the trainer selects the network at import time.
    import baseline_harness_fair4net as H
    import deq_trainer_fair4net as tr

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    setup_start = time.perf_counter()
    audit, meta, summary, data, hashes = validate_sources(args, tr, H)
    cfg, op, deq, checkpoint, physics_provenance = build_model(
        args, tr, H, data, summary, device
    )
    setup_elapsed = time.perf_counter() - setup_start
    test_groups = category_indices(data, 2)
    val_groups = category_indices(data, 1)
    counts = {name: int(values.size) for name, values in test_groups.items()}
    if not args.smoke and counts != {
        "pooled": args.expected_test,
        "clean": int(audit["category_counts"]["test"]["clean"]),
        "line": int(audit["category_counts"]["test"]["line"]),
        "generator": int(audit["category_counts"]["test"]["generator"]),
    }:
        raise ValueError(f"test category counts differ from audit: {counts}")

    first_arrays = batch_numpy(data, val_groups["pooled"][:1])
    d0, g0, l0 = to_device_batch(first_arrays, device)
    sync(device)
    first_start = time.perf_counter_ns()
    run_deq(deq, d0, g0, l0)
    sync(device)
    first_inference_ms = (time.perf_counter_ns() - first_start) / 1e6

    manifest = {
        "status": "CONTRACT_PASS" if args.contract_only else "RUNNING",
        "case": args.case, "protocol": EXPECTED_PROTOCOL,
        "timing_protocol": {
            "compatibility_source": "fair4net_line_forward_benchmark.py",
            "warmup": args.warmup, "repeats": args.repeats,
            "model_only": "CUDA events; inputs preloaded on GPU; DEQ solve included",
            "end_to_end": "synchronized wall time; CPU tensor preparation, H2D, DEQ, generator output D2H; excludes physics scoring and file I/O",
        },
        "solver": {
            "picard_iters": cfg.picard_iters, "anderson_m": cfg.anderson_m,
            "anderson_lam": cfg.anderson_lam, "anderson_max_iter": cfg.anderson_max_iter,
            "anderson_tol": cfg.anderson_tol,
        },
        "test_category_counts": counts,
        "hashes": hashes,
        "physics_provenance": physics_provenance,
        "setup_elapsed_sec": setup_elapsed,
        "cold_first_inference_ms": first_inference_ms,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_nK": float(checkpoint["nK"]),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
    }
    if args.contract_only:
        write_json(args.out_dir / "manifest.json", manifest)
        print(f"[mixed-inference] CONTRACT PASS case={args.case} counts={counts}")
        return

    capacity_rows, batch_sizes = discover_batches(
        deq, data, val_groups["pooled"], args.batch_sizes,
        args.safe_batch_candidates, device,
    )
    timing_rows = measure_timings(args, deq, op, data, test_groups, batch_sizes, device)
    aggregate_rows = aggregate_timing(timing_rows)
    accuracy_rows = evaluate_accuracy(args, H, op, deq, data, test_groups, device)
    manifest.update({
        "status": "PASS", "selected_timing_batch_sizes": batch_sizes,
        "batch_capacity_selected_on": "validation split only",
        "quality_batch_size": int(args.quality_batch_size or QUALITY_BATCH[args.case]),
        "environment": {
            "host": platform.node(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    })
    write_csv(args.out_dir / "raw_timing.csv", timing_rows)
    write_csv(args.out_dir / "aggregated_timing.csv", aggregate_rows)
    write_csv(args.out_dir / "accuracy_verification.csv", accuracy_rows)
    write_csv(args.out_dir / "batch_capacity_validation.csv", capacity_rows)
    write_json(args.out_dir / "environment_and_manifest.json", manifest)
    print(f"[mixed-inference] PASS case={args.case} batches={batch_sizes} out={args.out_dir}")


if __name__ == "__main__":
    main()
