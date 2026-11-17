#!/usr/bin/env python3
"""Generate preventive line-contingency SC-DCOPF labels for IEEE 300."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from phase_a_scopf.common import (
    CASE_NAME,
    DIAGNOSTIC_TIE_TOL_MW,
    FLOW_MODEL_CONTRACT,
    LINE_LIMIT_SCALE,
    NETWORK_SEED,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SECURITY_MODEL_CONTRACT,
    build_network_contract,
    discover_line_security_set,
    sample_pd_bus,
    select_smoke_security_set,
    sha256_file,
    solve_preventive_scopf,
    split_ids,
    static_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-low", type=float, default=0.9)
    parser.add_argument("--scale-high", type=float, default=1.1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--n-contingencies", type=int, default=32)
    parser.add_argument("--max-attempt-factor", type=float, default=3.0)
    args = parser.parse_args()
    if args.N < 1:
        raise ValueError("N must be positive")
    if not 0.0 < args.scale_low <= args.scale_high:
        raise ValueError("invalid demand scale range")

    started = time.perf_counter()
    _, _, static = build_network_contract()
    full_security = discover_line_security_set(static)
    full_set_started = time.perf_counter()
    try:
        full_set_solution = solve_preventive_scopf(
            static, full_security, static["base_pd"]
        )
    except RuntimeError as exc:
        if "linearly infeasible" not in str(exc):
            raise
        full_set_certificate = {
            "status": "INFEASIBLE",
            "method": "preventive constraint generation with independent HiGHS linear-feasibility oracle",
            "demand": "base_pd_plus_gs_bus",
            "eligible_contingencies": int(len(full_security["branch_rows"])),
            "failure": str(exc),
            "elapsed_sec": time.perf_counter() - full_set_started,
        }
    else:
        full_set_certificate = {
            "status": "FEASIBLE",
            "method": "preventive constraint generation with independent HiGHS linear-feasibility oracle",
            "demand": "base_pd_plus_gs_bus",
            "eligible_contingencies": int(len(full_security["branch_rows"])),
            "objective": float(full_set_solution.objective),
            "elapsed_sec": time.perf_counter() - full_set_started,
        }
    security = select_smoke_security_set(
        static, full_security, count=args.n_contingencies
    )
    print(
        json.dumps(
            {
                "event": "security_set",
                "case": CASE_NAME,
                "nbus": static["nbus"],
                "ngen": static["ngen"],
                "nbranch": static["nbranch"],
                "pandapower_lines": int(len(static["line_branch_rows"])),
                "eligible_non_islanding_line_outages": int(len(full_security["branch_rows"])),
                "declared_smoke_line_outages": int(len(security["branch_rows"])),
                "excluded_islanding_or_singular_line_outages": int(len(full_security["excluded"])),
                "screened_out_eligible_line_outages": int(len(security["screened_out_eligible"])),
                "selection_policy": security["selection_policy"],
                "security_set_digest": security["digest"],
                "full_eligible_set_certificate": full_set_certificate,
            }
        ),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    pd_rows = []
    pg_rows = []
    factor_rows = []
    objective_rows = []
    flow_rows = []
    worst_rows = []
    worst_cont_rows = []
    worst_branch_rows = []
    cut_round_rows = []
    active_cut_rows = []
    qp_iter_rows = []
    time_rows = []
    solver_success_rows = []
    solver_status_rows = []
    solver_message_rows = []
    balance_rows = []
    bound_violation_rows = []
    base_violation_rows = []
    source_rows = []
    failures = []
    warm = None
    attempts = 0
    max_attempts = max(args.N, int(np.ceil(args.max_attempt_factor * args.N)))
    while len(pd_rows) < args.N:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"accepted only {len(pd_rows)}/{args.N} samples after {attempts} attempts"
            )
        source_index = attempts
        attempts += 1
        pd_bus, factors = sample_pd_bus(
            static, rng, args.scale_low, args.scale_high
        )
        try:
            solved = solve_preventive_scopf(static, security, pd_bus, warm_start=warm)
        except Exception as exc:
            failures.append(
                {
                    "source_index": int(source_index),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if len(failures) <= 5 or len(failures) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "sample_reject",
                            "attempt": attempts,
                            "accepted": len(pd_rows),
                            "reason": str(exc),
                        }
                    ),
                    flush=True,
                )
            warm = None
            continue
        warm = solved.pg
        pd_rows.append(pd_bus)
        pg_rows.append(solved.pg)
        factor_rows.append(factors)
        objective_rows.append(solved.objective)
        flow_rows.append(solved.base_flow)
        worst_rows.append(solved.worst_violation_mw)
        worst_cont_rows.append(solved.worst_contingency_position)
        worst_branch_rows.append(solved.worst_branch_row)
        cut_round_rows.append(solved.cut_rounds)
        active_cut_rows.append(solved.active_security_rows)
        qp_iter_rows.append(solved.qp_iterations)
        time_rows.append(solved.solve_time_sec)
        solver_success_rows.append(solved.solver_success)
        solver_status_rows.append(solved.solver_status)
        solver_message_rows.append(solved.solver_message)
        balance_rows.append(solved.balance_abs_mw)
        bound_violation_rows.append(solved.generator_bound_violation_mw)
        base_violation_rows.append(solved.base_flow_violation_mw)
        source_rows.append(source_index)
        accepted = len(pd_rows)
        if accepted == args.N or accepted % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "sample_progress",
                        "accepted": accepted,
                        "target": args.N,
                        "attempts": attempts,
                        "rejected": attempts - accepted,
                        "latest_time_sec": solved.solve_time_sec,
                        "latest_cut_rounds": solved.cut_rounds,
                        "latest_active_security_rows": solved.active_security_rows,
                        "latest_worst_violation_mw": solved.worst_violation_mw,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.meta or args.out.with_suffix(".meta.json")
    arrays = {
        "dataset_schema_name": np.asarray(SCHEMA_NAME),
        "dataset_schema_version": np.int16(SCHEMA_VERSION),
        "case_name": np.asarray(CASE_NAME),
        "pd_bus": np.stack(pd_rows).astype(np.float64),
        "load_scale_factors": np.stack(factor_rows).astype(np.float64),
        "pg_scopf": np.stack(pg_rows).astype(np.float64),
        "base_flow": np.stack(flow_rows).astype(np.float64),
        "objective": np.asarray(objective_rows, dtype=np.float64),
        "split": split_ids(args.N),
        "source_index": np.asarray(source_rows, dtype=np.int64),
        "worst_security_violation_mw": np.asarray(worst_rows, dtype=np.float64),
        "worst_contingency_position": np.asarray(worst_cont_rows, dtype=np.int64),
        "worst_branch_row": np.asarray(worst_branch_rows, dtype=np.int64),
        "cut_rounds": np.asarray(cut_round_rows, dtype=np.int32),
        "active_security_rows": np.asarray(active_cut_rows, dtype=np.int32),
        "qp_iterations": np.asarray(qp_iter_rows, dtype=np.int32),
        "label_time_sec": np.asarray(time_rows, dtype=np.float64),
        "solver_success": np.asarray(solver_success_rows, dtype=np.uint8),
        "solver_status": np.asarray(solver_status_rows, dtype=np.int32),
        "solver_message": np.asarray(solver_message_rows, dtype=np.str_),
        "balance_abs_mw": np.asarray(balance_rows, dtype=np.float64),
        "generator_bound_violation_mw": np.asarray(
            bound_violation_rows, dtype=np.float64
        ),
        "base_flow_violation_mw": np.asarray(base_violation_rows, dtype=np.float64),
        "gs_bus": static["gs_bus"].astype(np.float64),
        "ptdf": static["ptdf"].astype(np.float64),
        "flow_offset": static["flow_offset"].astype(np.float64),
        "fmax": static["fmax"].astype(np.float64),
        "pmin": static["pmin"].astype(np.float64),
        "pmax": static["pmax"].astype(np.float64),
        "cost_c2": static["c2"].astype(np.float64),
        "cost_c1": static["c1"].astype(np.float64),
        "cost_c0": static["c0"].astype(np.float64),
        "bus_of_gen": static["bus_of_gen"].astype(np.int64),
        "security_branch_rows": security["branch_rows"].astype(np.int64),
        "security_line_pp_indices": security["line_pp_indices"].astype(np.int64),
        "security_lodf_columns": security["lodf_columns"].astype(np.float64),
        "pd_bus_includes_gs": np.int8(0),
        "diagnostic_tie_tolerance_mw": np.float64(DIAGNOSTIC_TIE_TOL_MW),
    }
    np.savez_compressed(args.out, **arrays)
    counts = np.bincount(arrays["split"], minlength=3)
    meta = {
        "dataset_schema_name": SCHEMA_NAME,
        "dataset_schema_version": SCHEMA_VERSION,
        "case_name": CASE_NAME,
        "N": args.N,
        "seed": args.seed,
        "network_seed": NETWORK_SEED,
        "line_limit_scale": LINE_LIMIT_SCALE,
        "demand_scale_range": [args.scale_low, args.scale_high],
        "split_counts": {
            "train": int(counts[0]),
            "validation": int(counts[1]),
            "test": int(counts[2]),
        },
        "flow_model_contract": FLOW_MODEL_CONTRACT,
        "security_model_contract": SECURITY_MODEL_CONTRACT,
        "demand_convention": "pd_bus is conventional demand; effective_demand=pd_bus+gs_bus",
        "contingency_set_shared_across_splits": True,
        "contingency_scope": "fixed declared subset of finite non-islanding pandapower net.line outages",
        "selection_policy": security["selection_policy"],
        "contingency_set_digest": security["digest"],
        "declared_contingencies": int(len(security["branch_rows"])),
        "eligible_non_islanding_contingencies": int(security["eligible_count"]),
        "excluded_islanding_or_singular_contingencies": security["excluded"],
        "screened_out_eligible_contingencies": security["screened_out_eligible"],
        "full_eligible_set_base_demand_certificate": full_set_certificate,
        "branch_limit_policy": "same scaled RATE_A limits in base and contingency states",
        "load_shedding_or_security_slack": False,
        "label_solver": {
            "primary": "SciPy SLSQP strictly convex QP with exact linear constraints",
            "linear_feasibility_oracle": "SciPy HiGHS",
            "security_separation_tolerance_mw": 1e-5,
            "diagnostic_identity_tie_tolerance_mw": DIAGNOSTIC_TIE_TOL_MW,
            "diagnostic_identity_tie_break": "lexicographic (state_position, branch_row), with base state_position=-1",
            "accepted_status_contract": "solver_success=1, solver_status=0, nonempty solver_message, all saved residual gates pass",
        },
        "dimensions": {
            "nbus": static["nbus"],
            "ngen": static["ngen"],
            "nbranch": static["nbranch"],
        },
        "phase_shift_nonzero_count": static["phase_shift_nonzero_count"],
        "static_hashes": static_hashes(static, security),
        "sampling": {
            "accepted": args.N,
            "attempts": attempts,
            "rejected": attempts - args.N,
            "failures": failures,
        },
        "label_metrics": {
            "worst_security_violation_abs_max_mw": float(np.max(worst_rows)),
            "label_time_mean_sec": float(np.mean(time_rows)),
            "label_time_p95_sec": float(np.percentile(time_rows, 95)),
            "label_time_total_sec": float(np.sum(time_rows)),
            "cut_rounds_max": int(np.max(cut_round_rows)),
            "active_security_rows_max": int(np.max(active_cut_rows)),
            "balance_abs_max_mw": float(np.max(balance_rows)),
            "generator_bound_violation_abs_max_mw": float(
                np.max(bound_violation_rows)
            ),
            "base_flow_violation_abs_max_mw": float(np.max(base_violation_rows)),
            "solver_success_rate": float(np.mean(solver_success_rows)),
            "solver_status_values": sorted(
                int(x) for x in np.unique(solver_status_rows).tolist()
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pandapower": importlib.metadata.version("pandapower"),
            "pypower": importlib.metadata.version("PYPOWER"),
        },
        "wall_time_sec": time.perf_counter() - started,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    source_paths = {
        "generator": Path(__file__).resolve(),
        "common": (Path(__file__).resolve().parent / "phase_a_scopf" / "common.py"),
        "package_init": (Path(__file__).resolve().parent / "phase_a_scopf" / "__init__.py"),
    }
    manifest = {
        "manifest_schema": "phase_a_scopf_generation_manifest_v2",
        "dataset": {"path": str(args.out), "sha256": sha256_file(args.out)},
        "metadata": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        "source": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "configuration": {
            "case": CASE_NAME,
            "N": args.N,
            "seed": args.seed,
            "demand_scale_range": [args.scale_low, args.scale_high],
            "declared_contingencies": int(len(security["branch_rows"])),
            "contingency_set_digest": security["digest"],
        },
        "environment": meta["environment"],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "GENERATED",
                "dataset": str(args.out),
                "metadata": str(meta_path),
                "manifest": str(manifest_path),
                "accepted": args.N,
                "rejected": attempts - args.N,
                "declared_contingencies": len(security["branch_rows"]),
                **meta["label_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
