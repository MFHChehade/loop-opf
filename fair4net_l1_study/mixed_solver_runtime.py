#!/usr/bin/env python3
"""DC-OPF runtime reference on stored mixed-contingency test rows.

The solver is the same pandapower DC-OPF path used to generate labels.  No
solver setting is tuned on test data.  Solve-only timing excludes network
copy/scenario preparation; end-to-end timing includes preparation and solve.
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np

from mixed_inference_utils import aggregate_timing


CASES = ("case300", "case1354pegase", "case2869pegase", "case9241pegase")
CATEGORIES = ("clean", "line", "generator")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generator_map(meta: dict) -> dict[int, dict]:
    records = {}
    for split_rows in meta["generator_candidate_pools"].values():
        for row in split_rows:
            records[int(row["cont_gen_idx"])] = row
    return records


def category_indices(split, line, gen, split_value=2):
    selected = split == int(split_value)
    if np.any((line >= 0) & (gen >= 0)):
        raise ValueError("simultaneous line and generator outage")
    return {
        "clean": np.flatnonzero(selected & (line < 0) & (gen < 0)),
        "line": np.flatnonzero(selected & (line >= 0) & (gen < 0)),
        "generator": np.flatnonzero(selected & (line < 0) & (gen >= 0)),
    }


def bus_lookup(net) -> np.ndarray:
    lookup = np.asarray(net._pd2ppc_lookups["bus"], dtype=np.int64)
    if lookup.ndim != 1:
        raise ValueError("unexpected pandapower bus lookup")
    return lookup


def prepare_net(base_net, target_pd_bus, line_pp_idx, gen_idx, gen_records, mixed):
    net = copy.deepcopy(base_net)
    lookup = bus_lookup(base_net)
    base_loads = np.asarray(base_net.load.p_mw, dtype=np.float64)
    pp_buses = np.asarray(base_net.load.bus, dtype=np.int64)
    internal = lookup[pp_buses]
    totals = np.bincount(internal, weights=base_loads, minlength=len(target_pd_bus))
    target = np.asarray(target_pd_bus, dtype=np.float64)
    factors = np.ones_like(base_loads)
    valid = totals[internal] > 0
    factors[valid] = target[internal[valid]] / totals[internal[valid]]
    net.load.loc[:, "p_mw"] = base_loads * factors
    if "q_mvar" in net.load.columns:
        net.load.loc[:, "q_mvar"] = np.asarray(base_net.load.q_mvar, dtype=np.float64) * factors
    if int(line_pp_idx) >= 0:
        net.line.at[int(line_pp_idx), "in_service"] = False
    if int(gen_idx) >= 0:
        record = gen_records.get(int(gen_idx))
        if record is None:
            raise KeyError(f"no generator metadata for cont_gen_idx={gen_idx}")
        mixed._apply_generator_outage(net, record)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--samples-per-category", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if os.environ.get("DEQ_CASE") != args.case:
        raise ValueError("set DEQ_CASE to --case before Python starts")
    if args.warmup != 1 or args.repeats != 3:
        raise ValueError("compatibility protocol requires warmup=1 and repeats=3")

    import deq_trainer_fair4net as tr
    import make_mixed_contingency_dataset as mixed

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("case_name") != args.case:
        raise ValueError("metadata case mismatch")
    with np.load(args.data, allow_pickle=False) as z:
        pd_bus = z["pd_bus"].astype(np.float32)
        pg = z["pg"].astype(np.float32)
        split = z["split"].astype(np.int64)
        line = z["cont_line_idx"].astype(np.int64)
        gen = z["cont_gen_idx"].astype(np.int64)
        if "cont_line_pp_idx" not in z.files:
            raise KeyError("dataset lacks cont_line_pp_idx required for solver reconstruction")
        line_pp = z["cont_line_pp_idx"].astype(np.int64)
    groups = category_indices(split, line, gen)
    gen_records = generator_map(meta)
    base_net = tr.build_base_net()
    # Build ppc/lookups outside all timed regions.
    base_ppc = tr.build_ppc_from_net(base_net)
    gs_total = float(np.asarray(tr.extract_static_mats(base_ppc)["gs_bus"], dtype=np.float64).sum())

    selected_count = 1 if args.smoke else args.samples_per_category
    raw_rows = []
    verify_rows = []
    for category in CATEGORIES:
        chosen = groups[category][:selected_count]
        if chosen.size != selected_count:
            raise ValueError(f"not enough {category} test rows")
        for sample_rank, idx in enumerate(chosen):
            idx = int(idx)
            def make_net():
                return prepare_net(
                    base_net, pd_bus[idx], line_pp[idx], gen[idx], gen_records, mixed
                )

            prepared = make_net()
            for _ in range(args.warmup):
                mixed.line_ds._solve_and_extract(copy.deepcopy(prepared))
            for repeat in range(args.repeats):
                fresh = copy.deepcopy(prepared)
                start = time.perf_counter_ns()
                sol = mixed.line_ds._solve_and_extract(fresh)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                raw_rows.append({
                    "case": args.case, "method": "dc_opf", "category": category,
                    "batch_size": 1, "timing_kind": "solver_only", "sample_rank": sample_rank,
                    "dataset_row": idx, "repeat": repeat, "latency_ms": elapsed_ms,
                    "peak_cuda_mem_alloc_bytes": 0,
                })
                verify_rows.append({
                    "case": args.case, "category": category, "dataset_row": idx,
                    "repeat": repeat,
                    "pg_max_abs_error_MW": float(np.max(np.abs(np.asarray(sol["pg"]) - pg[idx]))),
                    "balance_abs_MW": float(abs(np.asarray(sol["pg"], dtype=np.float64).sum()
                                               - np.asarray(sol["pd_bus"], dtype=np.float64).sum()
                                               - gs_total)),
                })

            for _ in range(args.warmup):
                mixed.line_ds._solve_and_extract(make_net())
            for repeat in range(args.repeats):
                start = time.perf_counter_ns()
                sol = mixed.line_ds._solve_and_extract(make_net())
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                raw_rows.append({
                    "case": args.case, "method": "dc_opf", "category": category,
                    "batch_size": 1, "timing_kind": "end_to_end", "sample_rank": sample_rank,
                    "dataset_row": idx, "repeat": repeat, "latency_ms": elapsed_ms,
                    "peak_cuda_mem_alloc_bytes": 0,
                })

    aggregate = aggregate_timing(raw_rows)
    # Weighted pooled rows use the authoritative test category counts.
    test_counts = {key: int(values.size) for key, values in groups.items()}
    for timing_kind in ("solver_only", "end_to_end"):
        rows = [r for r in aggregate if r["timing_kind"] == timing_kind]
        total = sum(test_counts[r["category"]] for r in rows)
        aggregate.append({
            "case": args.case, "method": "dc_opf", "category": "pooled_weighted",
            "batch_size": 1, "timing_kind": timing_kind,
            "repeats": sum(int(r["repeats"]) for r in rows),
            "latency_mean_ms": sum(r["latency_mean_ms"] * test_counts[r["category"]] for r in rows) / total,
            "latency_std_ms": None, "latency_p50_ms": None, "latency_p95_ms": None,
            "per_sample_p50_ms": None, "throughput_samples_per_sec": None,
            "peak_cuda_mem_alloc_bytes": 0,
        })

    if max(row["pg_max_abs_error_MW"] for row in verify_rows) > (0.1 if args.smoke else 0.05):
        raise ValueError("reconstructed solver output does not reproduce stored labels")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "solver_raw_timing.csv", raw_rows)
    write_csv(args.out_dir / "solver_aggregated_timing.csv", aggregate)
    write_csv(args.out_dir / "solver_reload_verification.csv", verify_rows)
    write_json(args.out_dir / "solver_manifest.json", {
        "status": "PASS", "case": args.case,
        "dataset_sha256": sha256_file(args.data), "metadata_sha256": sha256_file(args.meta),
        "warmup": args.warmup, "repeats": args.repeats,
        "samples_per_category": selected_count,
        "selection": "first deterministic stored test rows per category; no solver tuning",
        "solver_only": "prepared fresh network copy; timed DC-OPF solve and extraction",
        "end_to_end": "network copy, stored-demand reconstruction, contingency application, solve and extraction",
        "weighted_pooled_note": "weighted mean of category means; pooled p50/p95 intentionally undefined",
        "host": platform.node(), "python": platform.python_version(),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
    })
    print(f"[mixed-solver-runtime] PASS case={args.case} out={args.out_dir}")


if __name__ == "__main__":
    main()
