from __future__ import annotations

import math
import statistics
from typing import Iterable


CATEGORIES = ("pooled", "clean", "line", "generator")


def parse_positive_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise ValueError("expected comma-separated positive integers")
    return values


def category_name(cont_line_idx: int, cont_gen_idx: int) -> str:
    line = int(cont_line_idx)
    gen = int(cont_gen_idx)
    if line >= 0 and gen >= 0:
        raise ValueError("simultaneous line and generator outages are forbidden")
    if line >= 0:
        return "line"
    if gen >= 0:
        return "generator"
    return "clean"


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(q) / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def aggregate_timing(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["case"], row["method"], row["category"], int(row["batch_size"]),
            row["timing_kind"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(groups.items()):
        values = [float(row["latency_ms"]) for row in selected]
        batch = int(key[3])
        output.append({
            "case": key[0],
            "method": key[1],
            "category": key[2],
            "batch_size": batch,
            "timing_kind": key[4],
            "repeats": len(values),
            "latency_mean_ms": statistics.mean(values),
            "latency_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latency_p50_ms": statistics.median(values),
            "latency_p95_ms": percentile(values, 95),
            "per_sample_p50_ms": statistics.median(values) / batch,
            "throughput_samples_per_sec": 1000.0 * batch / statistics.median(values),
            "peak_cuda_mem_alloc_bytes": max(int(row.get("peak_cuda_mem_alloc_bytes", 0)) for row in selected),
        })
    return output
