#!/usr/bin/env python3
"""Exact fixed-K PDHG rollout ablation for the audited mixed N-1 campaign."""
from __future__ import annotations

import argparse
import csv
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


CASES = ("case300", "case1354pegase", "case2869pegase", "case9241pegase")
CATEGORIES = ("pooled", "clean", "line", "generator")
DEFAULT_K = (1, 2, 4, 8, 16, 32, 64, 128, 200, 256, 400, 600)
DEFAULT_BUDGET_MS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def parse_ints(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_floats(text: str) -> list[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive")
    return values


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def rollout_fixed_k(op, d, gen, line, k: int):
    """Apply the learned operator exactly k times from the zero state."""
    dim = int(op.ngen) + 1 + 2 * int(op.nbranch)
    z = torch.zeros((d.shape[0], dim), dtype=d.dtype, device=d.device)
    for _ in range(int(k)):
        z = op(z, d, gen, line)
    return z


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["case"], row["category"], int(row["k"]),
               int(row["batch_size"]), row["timing_kind"])
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        items = groups[key]
        values = [float(x["latency_ms"]) for x in items]
        batch = key[3]
        p50 = percentile(values, 50)
        output.append({
            "case": key[0], "method": "deq_fixed_k", "category": key[1],
            "k": key[2], "batch_size": batch, "timing_kind": key[4],
            "repetitions": len(values), "latency_mean_ms": statistics.mean(values),
            "latency_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latency_p50_ms": p50, "latency_p95_ms": percentile(values, 95),
            "per_sample_p50_ms": p50 / batch,
            "throughput_p50_samples_s": 1000.0 * batch / p50,
            "peak_cuda_mem_alloc_bytes": max(int(x["peak_cuda_mem_alloc_bytes"]) for x in items),
        })
    return output


def build_budget_rows(aggregated: list[dict[str, Any]], budgets: list[float]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in aggregated:
        key = (row["case"], row["category"], int(row["batch_size"]), row["timing_kind"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda x: int(x["k"]))
        envelope = 0.0
        points = []
        for item in items:
            observed = float(item["latency_p50_ms"])
            envelope = max(envelope, observed)
            points.append((int(item["k"]), observed, envelope))
        for budget in budgets:
            feasible = [p for p in points if p[2] <= budget]
            selected = feasible[-1] if feasible else None
            output.append({
                "case": key[0], "category": key[1], "batch_size": key[2],
                "timing_kind": key[3], "budget_ms": budget,
                "selected_k": selected[0] if selected else 0,
                "observed_p50_ms": selected[1] if selected else "",
                "conservative_monotone_p50_ms": selected[2] if selected else "",
            })
    return output


def fit_latency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["case"], row["category"], int(row["batch_size"]), row["timing_kind"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items()):
        x = np.asarray([float(r["k"]) for r in items])
        y = np.asarray([float(r["latency_p50_ms"]) for r in items])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        output.append({
            "case": key[0], "category": key[1], "batch_size": key[2],
            "timing_kind": key[3], "intercept_ms": float(intercept),
            "slope_ms_per_iteration": float(slope),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        })
    return output


def measure_timings(args, M, op, data, groups, device) -> list[dict[str, Any]]:
    rows = []
    for category in CATEGORIES:
        indices = groups[category]
        for batch_size in args.batch_sizes:
            if batch_size > indices.size:
                continue
            arrays = M.batch_numpy(data, indices[:batch_size])
            d, gen, line = M.to_device_batch(arrays, device)
            for k in args.k_values:
                for _ in range(args.warmup):
                    rollout_fixed_k(op, d, gen, line, k)
                sync(device)
                for repeat in range(args.repeats):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        z = rollout_fixed_k(op, d, gen, line, k)
                        end.record(); end.synchronize()
                        latency_ms = float(start.elapsed_time(end))
                        peak = int(torch.cuda.max_memory_allocated(device))
                    else:
                        start_ns = time.perf_counter_ns()
                        z = rollout_fixed_k(op, d, gen, line, k)
                        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
                        peak = 0
                    rows.append({
                        "case": args.case, "category": category, "k": k,
                        "batch_size": batch_size, "timing_kind": "model_only",
                        "repeat": repeat, "latency_ms": latency_ms,
                        "peak_cuda_mem_alloc_bytes": peak,
                    })

                for _ in range(args.warmup):
                    d2, g2, l2 = M.to_device_batch(arrays, device)
                    z2 = rollout_fixed_k(op, d2, g2, l2, k)
                    _ = z2[:, :int(op.ngen)].cpu()
                sync(device)
                for repeat in range(args.repeats):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    sync(device)
                    start_ns = time.perf_counter_ns()
                    d2, g2, l2 = M.to_device_batch(arrays, device)
                    z2 = rollout_fixed_k(op, d2, g2, l2, k)
                    _ = z2[:, :int(op.ngen)].cpu()
                    sync(device)
                    latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    rows.append({
                        "case": args.case, "category": category, "k": k,
                        "batch_size": batch_size, "timing_kind": "end_to_end",
                        "repeat": repeat, "latency_ms": latency_ms,
                        "peak_cuda_mem_alloc_bytes": peak,
                    })
    return rows


@torch.inference_mode()
def evaluate_quality(args, M, H, op, data, groups, device) -> list[dict[str, Any]]:
    pooled = groups["pooled"]
    if args.max_test_samples and args.max_test_samples < pooled.size:
        each = max(1, args.max_test_samples // 3)
        chosen = []
        for category in ("clean", "line", "generator"):
            chosen.extend(groups[category][:each].tolist())
        pooled = np.asarray(chosen[:args.max_test_samples], dtype=np.int64)

    predictions = {k: [] for k in args.k_values}
    residuals = {k: [] for k in args.k_values}
    demands, targets, gens, lines = [], [], [], []
    batch_size = int(args.quality_batch_size or M.QUALITY_BATCH[args.case])
    start_wall = time.perf_counter()
    k_set, max_k = set(args.k_values), max(args.k_values)
    for offset in range(0, pooled.size, batch_size):
        idx = pooled[offset:offset + batch_size]
        arrays = M.batch_numpy(data, idx)
        d, gen, line = M.to_device_batch(arrays, device)
        target = torch.as_tensor(data["pg"][idx], dtype=torch.float32, device=device)
        dim = int(op.ngen) + 1 + 2 * int(op.nbranch)
        z = torch.zeros((d.shape[0], dim), dtype=d.dtype, device=device)
        for step in range(1, max_k + 1):
            z = op(z, d, gen, line)
            if step in k_set:
                z_next = op(z, d, gen, line)
                alg = (z_next - z).norm(dim=1) / (1e-12 + z_next.norm(dim=1))
                predictions[step].append(z[:, :int(op.ngen)].clone())
                residuals[step].append(alg)
        demands.append(d); targets.append(target); gens.append(gen); lines.append(line)
    sync(device)
    elapsed = time.perf_counter() - start_wall
    d = torch.cat(demands); target = torch.cat(targets)
    gen = torch.cat(gens); line = torch.cat(lines)
    masks = {
        "pooled": torch.ones(d.shape[0], dtype=torch.bool, device=device),
        "clean": (line < 0) & (gen < 0),
        "line": (line >= 0) & (gen < 0),
        "generator": (line < 0) & (gen >= 0),
    }
    rows = []
    for k in args.k_values:
        p = torch.cat(predictions[k]); alg = torch.cat(residuals[k])
        for category in CATEGORIES:
            mask = masks[category]
            metrics = H.score(op, p[mask], d[mask], target[mask],
                              cont_line_idx=line[mask], cont_gen_idx=gen[mask])
            rows.append({
                "case": args.case, "method": "deq_fixed_k", "category": category,
                "k": k, "samples": int(mask.sum().item()),
                "rmse_MW": metrics["rmse_MW"],
                "cost_gap_abs_percent": 100.0 * metrics["cost_rel_gap_abs"],
                "viol_worst_mean_MW": metrics["viol_worst_mean_MW"],
                "viol_p95_MW": metrics["viol_p95_MW"],
                "viol_max_MW": metrics["viol_max_MW"],
                "flow_sat_rate": metrics["flow_sat_rate"],
                "bal_abs_MW": metrics["bal_abs_MW"],
                "bal_max_MW": metrics["bal_max_MW"],
                "bal_sat_rate": metrics["bal_sat_rate"],
                "fixed_point_residual_mean": float(alg[mask].mean().item()),
                "quality_sweep_elapsed_sec": elapsed if category == "pooled" else "",
            })
    return rows


def compare_full(quality: list[dict[str, Any]], reference: Path | None) -> list[dict[str, Any]]:
    if reference is None or not reference.is_file():
        return []
    with reference.open(newline="", encoding="utf-8") as handle:
        full = {r["category"]: r for r in csv.DictReader(handle)}
    metrics = ("rmse_MW", "cost_gap_abs_percent", "viol_worst_mean_MW",
               "flow_sat_rate", "bal_abs_MW")
    rows = []
    for row in quality:
        ref = full.get(str(row["category"]))
        if ref is None:
            continue
        out = {"case": row["case"], "category": row["category"], "k": row["k"]}
        for metric in metrics:
            out[f"fixed_k_{metric}"] = row[metric]
            out[f"full_deq_{metric}"] = float(ref[metric])
            out[f"delta_{metric}"] = float(row[metric]) - float(ref[metric])
        rows.append(out)
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    for name in ("data", "audit", "meta", "checkpoint", "summary", "out_dir"):
        ap.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    ap.add_argument("--full-reference-csv", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-values", type=parse_ints, default=list(DEFAULT_K))
    ap.add_argument("--batch-sizes", type=parse_ints, default=parse_ints("1,8,32,256"))
    ap.add_argument("--budgets-ms", type=parse_floats, default=list(DEFAULT_BUDGET_MS))
    ap.add_argument("--quality-batch-size", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--expected-test", type=int, default=2000)
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if os.environ.get("DEQ_CASE") != args.case:
        raise ValueError("set DEQ_CASE to --case before Python starts")
    if args.warmup != 1 or args.repeats != 3:
        raise ValueError("mixed-study compatibility protocol requires warmup=1 and repeats=3")
    import baseline_harness_fair4net as H
    import deq_trainer_fair4net as tr
    import mixed_deq_inference as M

    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    audit, meta, summary, data, hashes = M.validate_sources(args, tr, H)
    cfg, op, _deq, checkpoint, physics = M.build_model(args, tr, H, data, summary, device)
    test_groups = M.category_indices(data, 2)
    val_groups = M.category_indices(data, 1)
    counts = {k: int(v.size) for k, v in test_groups.items()}
    if not args.smoke and counts["pooled"] != args.expected_test:
        raise ValueError(f"expected {args.expected_test} test rows, found {counts}")

    capacity = []
    for batch_size in args.batch_sizes:
        idx = val_groups["pooled"][:batch_size]
        if idx.size != batch_size:
            raise ValueError(f"validation split has fewer than {batch_size} rows")
        arrays = M.batch_numpy(data, idx)
        d, gen, line = M.to_device_batch(arrays, device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        rollout_fixed_k(op, d, gen, line, max(args.k_values))
        sync(device)
        capacity.append({"batch_size": batch_size, "tested_k": max(args.k_values),
                         "status": "PASS", "split": "validation",
                         "peak_cuda_mem_alloc_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0})

    raw = measure_timings(args, M, op, data, test_groups, device)
    aggregated = aggregate_timing(raw)
    quality = evaluate_quality(args, M, H, op, data, test_groups, device)
    budgets = build_budget_rows(aggregated, args.budgets_ms)
    fits = fit_latency(aggregated)
    comparison = compare_full(quality, args.full_reference_csv)
    manifest = {
        "status": "PASS", "case": args.case,
        "study": "mixed_n1_exact_fixed_k_rollout", "initial_state": "zeros",
        "anderson_acceleration": False, "deq_root_solver_called": False,
        "operator": "checkpoint learned PDHG operator applied exactly K times",
        "k_values": args.k_values, "timing_batch_sizes": args.batch_sizes,
        "timing_protocol": {"warmup": args.warmup, "repeats": args.repeats,
                            "model_only": "synchronized CUDA events",
                            "end_to_end": "synchronized wall time including H2D and generator D2H"},
        "quality_test_rows": counts, "selection_split": "validation only for batch capacity",
        "hashes": hashes, "physics_provenance": physics,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_nK": float(checkpoint["nK"]),
        "full_reference_csv": str(args.full_reference_csv) if args.full_reference_csv else None,
        "environment": {"host": platform.node(), "python": platform.python_version(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "command": " ".join(shlex.quote(x) for x in sys.argv),
    }
    write_csv(args.out_dir / "raw_fixed_k_timing.csv", raw)
    write_csv(args.out_dir / "aggregated_fixed_k_timing.csv", aggregated)
    write_csv(args.out_dir / "quality_by_k.csv", quality)
    write_csv(args.out_dir / "budget_to_k.csv", budgets)
    write_csv(args.out_dir / "latency_fit.csv", fits)
    write_csv(args.out_dir / "batch_capacity_validation.csv", capacity)
    if comparison:
        write_csv(args.out_dir / "fixed_k_vs_full_deq.csv", comparison)
    write_json(args.out_dir / "fixed_k_manifest.json", manifest)
    print(f"[mixed-fixed-k] PASS case={args.case} K={args.k_values} out={args.out_dir}")


if __name__ == "__main__":
    main()
