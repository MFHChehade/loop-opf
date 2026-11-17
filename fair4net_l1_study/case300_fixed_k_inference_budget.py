#!/usr/bin/env python3
"""Fixed-depth inference-budget study for preserved scalar L1-DEQ models.

The trained PDHG operator is rolled out exactly K times from the zero state.
Anderson acceleration and tolerance-based early stopping are intentionally not
used.  The program measures latency after warm-up and evaluates the common
fair4net cost, RMSE, balance, and branch-flow metrics on the fixed test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

os.environ.setdefault("DEQ_CASE", "case300")

import baseline_harness_fair4net as H  # noqa: E402
import deq_trainer_fair4net as tr  # noqa: E402


EXPECTED_ARTIFACTS = {
    "case300": {
        "dataset_sha256": "4a43cf19f232c34d3537a54b3883d89b770c7ed3275e6c581adb1c7ae11597f4",
        "checkpoint_sha256": "604ea5b6da4b220e269bcd3a4daa80042f6a5322d44ae8ae66c9db057987febe",
        "source_summary_sha256": "5ce1117525c8bb602a3f56da190a8dcfabc033c5c3eadf3257e731096e23284d",
    },
    "case1354pegase": {
        "dataset_sha256": "66a8eae91aac76b84576974d060b3c5ee23bc311c9913941c1d00510525647c7",
        "checkpoint_sha256": "02432c2072aece2259b28ccba406f385f637d64305e70fccc214d79d336f8e51",
        "source_summary_sha256": "6de5daa8cfb7652bcdd9f20bc1fc699d22b16fdb7db8ea214d02d2478016e6a6",
    },
    "case2869pegase": {
        "dataset_sha256": "ff4cd228e2ceef374d4391eed1c46d453c312e8f2257adbc8d4cf14d9dfbecb0",
        "checkpoint_sha256": "8b56d0f7667f67ffcbb8d83ee17ce0790787c82a09d7ac7f2dd2aa0609e1b51f",
        "source_summary_sha256": "002efa572a1cb094d45c3f0a4aacf9b6a9933ad302a1b83df810b4b593b2dd42",
    },
}
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
FLOW_TOL_MW = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=np.float64), q))


@torch.inference_mode()
def rollout_fixed_k(
    op: tr.PDHGOperatorLearnable,
    d_bus: torch.Tensor,
    cont_gen_idx: torch.Tensor | None,
    k: int,
) -> torch.Tensor:
    if k < 1:
        raise ValueError("K must be at least 1")
    dim = int(op.ngen) + 1 + 2 * int(op.nbranch)
    z = torch.zeros(d_bus.shape[0], dim, dtype=d_bus.dtype, device=d_bus.device)
    for _ in range(k):
        z = op(z, d_bus, cont_gen_idx)
    return z


def validate_source(
    data_path: Path,
    audit_path: Path,
    checkpoint_path: Path,
    source_summary_path: Path,
    case_name: str,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    for path in (data_path, audit_path, checkpoint_path, source_summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    hashes = {
        "dataset_sha256": sha256_file(data_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_summary_sha256": sha256_file(source_summary_path),
        "audit_sha256": sha256_file(audit_path),
    }
    expected = EXPECTED_ARTIFACTS.get(case_name)
    if expected is None:
        raise ValueError(f"unsupported production case: {case_name}")
    if not smoke:
        for key in ("dataset_sha256", "checkpoint_sha256", "source_summary_sha256"):
            if hashes[key] != expected[key]:
                raise ValueError(f"{key} is not the preserved {case_name} production artifact")
    audit = read_json(audit_path)
    source_summary = read_json(source_summary_path)
    if audit.get("case") != case_name or audit.get("protocol") != "fair4net_l1_v2_phase_aware":
        raise ValueError(f"the {case_name} phase-aware audit is required")
    if not smoke and audit.get("output_sha256") != hashes["dataset_sha256"]:
        raise ValueError("dataset hash differs from the audit")
    if source_summary.get("case_name") != case_name:
        raise ValueError(f"source summary is not {case_name}")
    source_cfg = dict(source_summary.get("config") or {})
    if source_cfg.get("step_mode") != "scalar" or float(source_cfg.get("w_flow_l1", 0.0)) <= 0.0:
        raise ValueError("source summary is not the selected scalar L1 model")
    return audit, source_summary, hashes


def build_operator_and_data(
    data_path: Path,
    audit: dict[str, Any],
    source_summary: dict[str, Any],
    checkpoint_path: Path,
    case_name: str,
    device: torch.device,
    smoke: bool,
) -> tuple[tr.PDHGOperatorLearnable, dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = H.load_gencont(str(data_path))
    if data["pg"] is None or data["split"] is None:
        raise ValueError("the fixed-K study requires solver labels and the saved split")
    if data.get("cont_line") is not None and np.any(np.asarray(data["cont_line"]) >= 0):
        if not smoke:
            raise ValueError(f"line-outage data cannot be used for the non-line {case_name} checkpoint")
        print("[fixed-k] SMOKE ONLY: ignoring line-outage metadata in the local fixture")

    net = tr.build_base_net()
    ppc = tr.build_ppc_from_net(net)
    static_np = tr.extract_static_mats(ppc)
    static_np["lodf"] = np.zeros(
        (int(static_np["nbranch"]), int(static_np["nbranch"])), dtype=np.float32
    )
    H.validate_contract(data, static_np, require_metadata=True)
    for key in ("gs_sha256", "fmax_sha256", "bus_of_gen_sha256"):
        if tr.sha256_array(static_np[{"gs_sha256": "gs_bus", "fmax_sha256": "fmax", "bus_of_gen_sha256": "bus_of_gen"}[key]]) != audit.get(key):
            raise ValueError(f"physics contract mismatch: {key}")
    runtime_flow_hash = tr.sha256_array(static_np["flow_offset"])
    source_flow_hash = (
        (source_summary.get("physics") or {}).get("flow_offset_sha256")
        or audit.get("flow_offset_sha256")
    )
    accepted_thread_equivalent = False
    if runtime_flow_hash != source_flow_hash:
        message = (
            f"runtime flow-offset hash {runtime_flow_hash} differs from the checkpoint "
            f"summary hash {source_flow_hash}"
        )
        observed_pair = (runtime_flow_hash, source_flow_hash)
        if observed_pair == THREAD_EQUIVALENT_FLOW_HASH_PAIRS.get(case_name):
            accepted_thread_equivalent = True
            print(
                f"[fixed-k] accepted provenance-audited thread-equivalent "
                f"flow-offset hashes for {case_name}: {observed_pair}"
            )
        elif not smoke:
            raise ValueError(message)
        else:
            print(f"[fixed-k] SMOKE ONLY: {message}")
    if abs(float(audit.get("line_limit_scale", -1.0)) - float(tr.LINE_LIMIT_SCALE)) > 1e-12:
        raise ValueError("line-limit convention mismatch")

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("case_name") != case_name:
        raise ValueError(f"checkpoint is not {case_name}")
    old_cfg = dict(checkpoint.get("cfg") or {})
    if old_cfg.get("step_mode") != "scalar" or float(old_cfg.get("w_flow_l1", 0.0)) <= 0.0:
        raise ValueError("checkpoint is not the selected scalar L1 model")
    if "steps_state" not in checkpoint or "nK" not in checkpoint:
        raise ValueError("checkpoint lacks learned step state")

    cfg = tr.TrainCfg(device=str(device))
    static = tr.to_torch_static(static_np, cfg)
    steps = tr.LearnableSteps(
        nK=float(checkpoint["nK"]),
        s_max=float(old_cfg.get("s_max", cfg.s_max)),
        rho_init=0.9,
        s_init=0.0,
    ).to(device)
    steps.load_state_dict(checkpoint["steps_state"], strict=True)
    op = tr.PDHGOperatorLearnable(static, steps).to(device)
    op.eval()
    flow_provenance = {
        "runtime_flow_offset_sha256": runtime_flow_hash,
        "audit_flow_offset_sha256": audit.get("flow_offset_sha256"),
        "checkpoint_summary_flow_offset_sha256": source_flow_hash,
        "accepted_thread_equivalent_pair": accepted_thread_equivalent,
        "limitation": (
            "The audit/runtime and checkpoint-summary flow-offset hashes are an exact "
            "pair whitelisted by the established project validators; the historical raw "
            "checkpoint-summary offset array is unavailable for independent equality checking."
            if accepted_thread_equivalent else None
        ),
    }
    return op, data, checkpoint, flow_provenance


@torch.inference_mode()
def timed_batch(
    op: tr.PDHGOperatorLearnable,
    d_bus: torch.Tensor,
    cont: torch.Tensor | None,
    k: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        rollout_fixed_k(op, d_bus, cont, k)
    synchronize(device)
    elapsed_ms: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rollout_fixed_k(op, d_bus, cont, k)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
        else:
            start_cpu = time.perf_counter_ns()
            rollout_fixed_k(op, d_bus, cont, k)
            elapsed_ms.append((time.perf_counter_ns() - start_cpu) / 1e6)
    return elapsed_ms


@torch.inference_mode()
def evaluate_quality(
    op: tr.PDHGOperatorLearnable,
    data: dict[str, Any],
    indices: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    demands: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    synchronize(device)
    start = time.perf_counter()
    for offset in range(0, indices.size, batch_size):
        idx = indices[offset : offset + batch_size]
        d = torch.as_tensor(data["pd_bus"][idx], dtype=torch.float32, device=device)
        target = torch.as_tensor(data["pg"][idx], dtype=torch.float32, device=device)
        cont = None
        if data["cont"] is not None:
            cont = torch.as_tensor(data["cont"][idx], dtype=torch.long, device=device)
        z = rollout_fixed_k(op, d, cont, k)
        z_next = op(z, d, cont)
        residuals.append((z_next - z).norm(dim=1) / (1e-12 + z_next.norm(dim=1)))
        predictions.append(z[:, : int(op.ngen)])
        demands.append(d)
        targets.append(target)
    synchronize(device)
    elapsed = time.perf_counter() - start
    p_all = torch.cat(predictions)
    d_all = torch.cat(demands)
    target_all = torch.cat(targets)
    metrics = H.score(op, p_all, d_all, target_all)
    metrics["cost_gap_abs_percent"] = 100.0 * metrics["cost_rel_gap_abs"]
    metrics["fixed_point_residual_mean"] = float(torch.cat(residuals).mean())
    metrics["quality_pass_elapsed_ms"] = 1000.0 * elapsed
    metrics["quality_pass_samples_per_sec"] = float(indices.size / elapsed)
    return metrics


def fit_latency(rows: list[dict[str, Any]], batch_size: int) -> dict[str, float]:
    selected = [r for r in rows if int(r["batch_size"]) == batch_size]
    x = np.asarray([r["k"] for r in selected], dtype=np.float64)
    y = np.asarray([r["latency_median_ms"] for r in selected], dtype=np.float64)
    if x.size == 1:
        return {
            "batch_size": int(batch_size),
            "intercept_ms": float(y[0]),
            "slope_ms_per_layer": 0.0,
            "r_squared": 1.0,
            "calibration_points": 1,
        }
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom if denom > 0 else 1.0
    return {
        "batch_size": int(batch_size),
        "intercept_ms": float(intercept),
        "slope_ms_per_layer": float(slope),
        "r_squared": float(r2),
        "calibration_points": int(x.size),
    }


def parse_int_list(value: str) -> list[int]:
    out = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    if not out or out[0] < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def parse_float_list(value: str) -> list[float]:
    out = sorted(set(float(x.strip()) for x in value.split(",") if x.strip()))
    if not out or out[0] <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-name", default=os.environ.get("DEQ_CASE", "case300"), choices=sorted(EXPECTED_ARTIFACTS))
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--source-stem", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-values", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32,64,128,200,256"))
    ap.add_argument("--timing-batch-sizes", type=parse_int_list, default=parse_int_list("1,16"))
    ap.add_argument("--budgets-ms", type=parse_float_list, default=parse_float_list("1,2,5,10,20,50,100"))
    ap.add_argument("--quality-batch-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timing-repeats", type=int, default=30)
    ap.add_argument("--max-test-samples", type=int, default=0, help="0 means the complete fixed test split")
    ap.add_argument("--smoke", action="store_true", help="permit non-production data and a reduced test subset")
    ap.add_argument("--contract-only", action="store_true", help="validate immutable artifacts and rebuilt physics, then exit")
    args = ap.parse_args()

    if tr.CASE_NAME != args.case_name:
        raise ValueError(
            f"DEQ_CASE selected {tr.CASE_NAME}, but --case-name requested {args.case_name}; "
            "set DEQ_CASE before launching Python"
        )
    if args.quality_batch_size < 1 or args.warmup < 0 or args.timing_repeats < 1:
        raise ValueError("batch size/repeat arguments are invalid")
    source_ckpt = Path(str(args.source_stem) + ".pt")
    source_summary_path = Path(str(args.source_stem) + "_summary.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    audit, source_summary, hashes = validate_source(
        args.data, args.audit, source_ckpt, source_summary_path, args.case_name, args.smoke
    )
    op, data, checkpoint, flow_provenance = build_operator_and_data(
        args.data, audit, source_summary, source_ckpt, args.case_name, device, args.smoke
    )
    test_idx = np.flatnonzero(np.asarray(data["split"], dtype=np.int64) == 2)
    if not args.smoke and test_idx.size != 2000:
        raise ValueError(f"expected 2000 fixed test samples, found {test_idx.size}")
    if args.contract_only:
        print(
            f"[fixed-k] CONTRACT PASS: case={args.case_name} test_samples={test_idx.size} "
            f"nbus={int(op.nbus)} ngen={int(op.ngen)} nbranch={int(op.nbranch)}"
        )
        return
    if args.max_test_samples > 0:
        test_idx = test_idx[: args.max_test_samples]
    if test_idx.size < max(args.timing_batch_sizes):
        raise ValueError("test subset is smaller than a requested timing batch")

    timing_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for k in args.k_values:
        for batch_size in args.timing_batch_sizes:
            idx = test_idx[:batch_size]
            d = torch.as_tensor(data["pd_bus"][idx], dtype=torch.float32, device=device)
            cont = None if data["cont"] is None else torch.as_tensor(data["cont"][idx], dtype=torch.long, device=device)
            samples = timed_batch(op, d, cont, k, args.warmup, args.timing_repeats, device)
            median_ms = float(statistics.median(samples))
            timing_rows.append({
                "k": int(k),
                "batch_size": int(batch_size),
                "latency_mean_ms": float(statistics.mean(samples)),
                "latency_median_ms": median_ms,
                "latency_p05_ms": percentile(samples, 5),
                "latency_p95_ms": percentile(samples, 95),
                "per_sample_median_ms": median_ms / batch_size,
                "throughput_samples_per_sec": 1000.0 * batch_size / median_ms,
                "warmup_runs": int(args.warmup),
                "timed_repeats": int(args.timing_repeats),
            })
        metrics = evaluate_quality(op, data, test_idx, k, args.quality_batch_size, device)
        quality_rows.append({"k": int(k), "test_samples": int(test_idx.size), **metrics})
        print(
            f"[fixed-k] K={k:3d} rmse={metrics['rmse_MW']:.6g} MW "
            f"cost_gap={metrics['cost_gap_abs_percent']:.6g}% "
            f"viol={metrics['viol_worst_mean_MW']:.6g} MW "
            f"balance={metrics['bal_abs_MW']:.6g} MW"
        )

    fits = [fit_latency(timing_rows, b) for b in args.timing_batch_sizes]
    timing_lookup = {
        (int(row["k"]), int(row["batch_size"])): row for row in timing_rows
    }
    for quality in quality_rows:
        for batch_size in args.timing_batch_sizes:
            measured = timing_lookup[(int(quality["k"]), batch_size)]
            quality[f"latency_b{batch_size}_median_ms"] = measured["latency_median_ms"]
            quality[f"latency_b{batch_size}_p95_ms"] = measured["latency_p95_ms"]

    quality_lookup = {int(row["k"]): row for row in quality_rows}
    budget_rows: list[dict[str, Any]] = []
    for batch_size in args.timing_batch_sizes:
        selected = [r for r in timing_rows if int(r["batch_size"]) == batch_size]
        for budget in args.budgets_ms:
            feasible = [int(r["k"]) for r in selected if float(r["latency_median_ms"]) <= budget]
            selected_k = max(feasible) if feasible else None
            quality = quality_lookup.get(selected_k, {})
            budget_rows.append({
                "batch_size": int(batch_size),
                "budget_ms": float(budget),
                "max_measured_k": selected_k,
                "rmse_MW": quality.get("rmse_MW"),
                "cost_gap_abs_percent": quality.get("cost_gap_abs_percent"),
                "viol_worst_mean_MW": quality.get("viol_worst_mean_MW"),
                "viol_p95_MW": quality.get("viol_p95_MW"),
                "viol_max_MW": quality.get("viol_max_MW"),
                "flow_sat_rate": quality.get("flow_sat_rate"),
                "bal_abs_MW": quality.get("bal_abs_MW"),
                "bal_max_MW": quality.get("bal_max_MW"),
                "bal_sat_rate": quality.get("bal_sat_rate"),
                "fixed_point_residual_mean": quality.get("fixed_point_residual_mean"),
            })

    metadata = {
        "experiment": f"{args.case_name}_fixed_k_inference_budget_v1",
        "case_name": args.case_name,
        "rollout": "z0=0; z_{j+1}=T_theta(z_j,d,cont_gen_idx); exactly K steps; no Anderson; no early stopping",
        "metric_contract": "common fair4net phase-aware scorer; cost gap absolute relative percent; flow and balance tolerance 0.001 MW",
        "flow_tolerance_MW": FLOW_TOL_MW,
        "smoke": bool(args.smoke),
        "device": str(device),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "k_values": args.k_values,
        "timing_batch_sizes": args.timing_batch_sizes,
        "quality_batch_size": int(args.quality_batch_size),
        "test_samples": int(test_idx.size),
        "hashes": hashes,
        "flow_offset_provenance": flow_provenance,
        "checkpoint_best_epoch": int(source_summary.get("best_epoch", -1)),
        "learned_steps": dict(source_summary.get("steps") or {}),
        "converged_reference_metrics": dict(source_summary.get("test") or {}),
        "checkpoint_nK": float(checkpoint["nK"]),
        "latency_linear_fits": fits,
        "peak_cuda_mem_alloc_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    prefix = f"{args.case_name}_fixed_k"
    write_csv(args.out_dir / f"{prefix}_timing.csv", timing_rows)
    write_csv(args.out_dir / f"{prefix}_quality.csv", quality_rows)
    write_csv(args.out_dir / f"{args.case_name}_budget_to_k.csv", budget_rows)
    atomic_json(args.out_dir / f"{prefix}_results.json", {
        "metadata": metadata,
        "timing": timing_rows,
        "quality": quality_rows,
        "budget_to_k": budget_rows,
    })
    atomic_json(args.out_dir / f"{prefix}_manifest.json", metadata)
    print(f"[fixed-k] PASS: {args.out_dir / f'{prefix}_results.json'}")


if __name__ == "__main__":
    main()
