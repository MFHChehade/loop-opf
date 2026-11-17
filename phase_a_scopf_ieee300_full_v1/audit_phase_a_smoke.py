#!/usr/bin/env python3
"""Independent float64, direct-topology, PyTorch, and KKT audit.

The auditor intentionally does not call the label solver or its contingency
flow helper.  It rebuilds every declared outage topology from the raw PPC,
derives independent post-outage PTDF/phase-offset operators, and certifies the
strictly convex labels with primal feasibility plus KKT stationarity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linprog, lsq_linear
from scipy.sparse import csr_matrix, diags, vstack

from phase_a_scopf.common import (
    CASE_NAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    _pypower_imports,
    build_network_contract,
    sha256_file,
    stable_json_digest,
    static_hashes,
)


def _dense_1d(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _direct_offset(make_bdc, base_mva, bus, branch, ptdf) -> np.ndarray:
    try:
        built = make_bdc(base_mva, bus, branch)
    except TypeError:
        built = make_bdc(bus, branch)
    pbusinj = _dense_1d(built[2])
    pfinj = _dense_1d(built[3])
    return base_mva * (pfinj - ptdf @ pbusinj)


def _row_digest(row: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()


def _independent_security_sets(
    static: Dict[str, Any], declared_count: int, denominator_tol: float = 1e-8
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Re-derive eligibility, LODFs, and severity selection without generator helpers."""
    transfer = (
        static["ptdf"][:, static["branch_from"]]
        - static["ptdf"][:, static["branch_to"]]
    )
    valid_rows: List[int] = []
    valid_pp: List[int] = []
    columns: List[np.ndarray] = []
    excluded: List[Dict[str, Any]] = []
    for pp_idx, row in zip(static["line_pp_indices"], static["line_branch_rows"]):
        denominator = float(1.0 - transfer[int(row), int(row)])
        if not np.isfinite(denominator) or abs(denominator) <= denominator_tol:
            excluded.append(
                {
                    "line_pp_idx": int(pp_idx),
                    "branch_row": int(row),
                    "denominator": denominator,
                }
            )
            continue
        column = np.asarray(transfer[:, int(row)] / denominator, dtype=np.float64)
        column[int(row)] = -1.0
        if not np.all(np.isfinite(column)):
            raise ValueError(f"nonfinite independently derived LODF for row {row}")
        valid_pp.append(int(pp_idx))
        valid_rows.append(int(row))
        columns.append(column)
    if declared_count < 1 or declared_count > len(valid_rows):
        raise ValueError("declared contingency count is outside independent eligible set")
    full = {
        "line_pp_indices": np.asarray(valid_pp, dtype=np.int64),
        "branch_rows": np.asarray(valid_rows, dtype=np.int64),
        "lodf_columns": np.stack(columns, axis=1),
        "excluded": excluded,
    }
    effective = static["base_pd"] + static["gs_bus"]
    base_flow = (
        static["ptdf"][:, static["bus_of_gen"]] @ static["base_pg"]
        + static["flow_offset"]
        - static["ptdf"] @ effective
    )
    score = np.abs(base_flow[full["branch_rows"]])
    selected = np.argsort(-score, kind="stable")[:declared_count]
    rows = full["branch_rows"][selected]
    pp_indices = full["line_pp_indices"][selected]
    payload = [
        {"line_pp_idx": int(pp), "branch_row": int(row)}
        for pp, row in zip(pp_indices, rows)
    ]
    declared = {
        "line_pp_indices": pp_indices,
        "branch_rows": rows,
        "lodf_columns": full["lodf_columns"][:, selected],
        "digest": stable_json_digest(payload),
    }
    return full, declared


def _graph_bridge_rows(static: Dict[str, Any]) -> np.ndarray:
    """Find line outages that disconnect the full line-plus-transformer graph."""
    nbus = int(static["nbus"])
    all_rows = np.arange(static["nbranch"], dtype=np.int64)

    def component_count(removed: int) -> int:
        parent = np.arange(nbus, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for row in all_rows:
            if int(row) == removed:
                continue
            union(int(static["branch_from"][row]), int(static["branch_to"][row]))
        return len({find(bus) for bus in range(nbus)})

    base_components = component_count(-1)
    if base_components != 1:
        raise ValueError(f"base internal network has {base_components} components")
    bridges = [
        int(row)
        for row in static["line_branch_rows"]
        if component_count(int(row)) > base_components
    ]
    return np.asarray(bridges, dtype=np.int64)


def _independent_full_set_feasibility(
    static: Dict[str, Any], full: Dict[str, Any], cuts_per_round: int = 128
) -> Dict[str, Any]:
    """Pure LP separation certificate for the full preventive outage intersection."""
    started = time.perf_counter()
    effective = static["base_pd"] + static["gs_bus"]
    demand_total = float(effective.sum())
    gen_flow = static["ptdf"][:, static["bus_of_gen"]]
    constant = static["flow_offset"] - static["ptdf"] @ effective
    finite = np.flatnonzero(static["fmax"] < 1e8)
    matrices = [csr_matrix(np.ones((1, static["ngen"]))), csr_matrix(gen_flow[finite])]
    lower = [np.asarray([demand_total]), -static["fmax"][finite] - constant[finite]]
    upper = [np.asarray([demand_total]), static["fmax"][finite] - constant[finite]]
    active: set[Tuple[int, int]] = set()
    history: List[Dict[str, Any]] = []
    for round_idx in range(32):
        matrix = vstack(matrices, format="csr")
        lo = np.concatenate(lower)
        hi = np.concatenate(upper)
        flow_a = matrix[1:]
        result = linprog(
            np.zeros(static["ngen"], dtype=np.float64),
            A_ub=vstack([flow_a, -flow_a], format="csr"),
            b_ub=np.concatenate([hi[1:], -lo[1:]]),
            A_eq=matrix[:1],
            b_eq=np.asarray([demand_total]),
            bounds=list(zip(static["pmin"], static["pmax"])),
            method="highs",
        )
        if not bool(result.success):
            return {
                "status": "INFEASIBLE",
                "method": "independent HiGHS LP constraint generation",
                "infeasible_round": round_idx,
                "active_security_pairs": len(active),
                "solver_status": int(result.status),
                "solver_message": str(result.message),
                "history": history,
                "elapsed_sec": time.perf_counter() - started,
            }
        pg = np.asarray(result.x, dtype=np.float64)
        base_flow = gen_flow @ pg + constant
        contingency = (
            base_flow[None, :]
            + full["lodf_columns"].T
            * base_flow[full["branch_rows"], None]
        )
        violation = np.abs(contingency) - static["fmax"][None, :]
        violation[np.arange(len(full["branch_rows"])), full["branch_rows"]] = -np.inf
        max_violation = float(np.max(violation))
        history.append(
            {
                "round": round_idx,
                "active_security_pairs": len(active),
                "max_omitted_violation_mw": max(0.0, max_violation),
            }
        )
        if max_violation <= 1e-7:
            return {
                "status": "FEASIBLE",
                "method": "independent HiGHS LP constraint generation",
                "round": round_idx,
                "active_security_pairs": len(active),
                "history": history,
                "elapsed_sec": time.perf_counter() - started,
            }
        extra_rows = []
        extra_lower = []
        extra_upper = []
        for flat in np.argsort(violation, axis=None)[::-1]:
            cont_pos, branch_row = np.unravel_index(int(flat), violation.shape)
            if float(violation[cont_pos, branch_row]) <= 1e-7:
                break
            pair = (int(cont_pos), int(branch_row))
            if pair in active:
                continue
            outage_row = int(full["branch_rows"][cont_pos])
            lodf_value = float(full["lodf_columns"][branch_row, cont_pos])
            row = gen_flow[branch_row] + lodf_value * gen_flow[outage_row]
            offset = constant[branch_row] + lodf_value * constant[outage_row]
            extra_rows.append(row)
            extra_lower.append(-static["fmax"][branch_row] - offset)
            extra_upper.append(static["fmax"][branch_row] - offset)
            active.add(pair)
            if len(extra_rows) >= cuts_per_round:
                break
        if not extra_rows:
            raise ValueError("full-set LP separation stalled with a positive violation")
        matrices.append(csr_matrix(np.asarray(extra_rows)))
        lower.append(np.asarray(extra_lower))
        upper.append(np.asarray(extra_upper))
    raise ValueError("full-set LP certificate exceeded the deterministic round limit")


def _direct_outage_operators(
    ppc: Dict[str, Any],
    static: Dict[str, Any],
    branch_rows: np.ndarray,
    progress_every: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    imp = _pypower_imports()
    h_list: List[np.ndarray] = []
    phi_list: List[np.ndarray] = []
    started = time.perf_counter()
    for pos, outage_row in enumerate(branch_rows):
        trial = copy.deepcopy(ppc)
        trial["branch"][int(outage_row), imp["BR_STATUS"]] = 0.0
        trial_i = imp["ext2int"](trial)
        bus = np.asarray(trial_i["bus"], dtype=np.float64)
        branch = np.asarray(trial_i["branch"], dtype=np.float64)
        if bus.shape[0] != static["nbus"]:
            raise ValueError(f"outage row {outage_row} changed the internal bus count")
        if branch.shape[0] != static["nbranch"] - 1:
            raise ValueError(f"outage row {outage_row} did not remove exactly one branch")
        ref_rows = np.flatnonzero(bus[:, imp["BUS_TYPE"]] == imp["REF"])
        ref = int(ref_rows[0]) if len(ref_rows) else 0
        h_small = np.asarray(
            imp["makePTDF"](float(trial_i["baseMVA"]), bus, branch, ref),
            dtype=np.float64,
        )
        phi_small = _direct_offset(
            imp["makeBdc"], float(trial_i["baseMVA"]), bus, branch, h_small
        )
        keep = np.delete(np.arange(static["nbranch"], dtype=np.int64), int(outage_row))
        h_full = np.zeros_like(static["ptdf"], dtype=np.float64)
        phi_full = np.zeros(static["nbranch"], dtype=np.float64)
        h_full[keep] = h_small
        phi_full[keep] = phi_small
        h_list.append(h_full)
        phi_list.append(phi_full)
        if (pos + 1) == len(branch_rows) or (pos + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "direct_topology_progress",
                        "validated": pos + 1,
                        "target": len(branch_rows),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return h_list, phi_list, {"elapsed_sec": time.perf_counter() - started}


def _alternative_qp_crosscheck(
    static: Dict[str, Any],
    pd_bus: np.ndarray,
    saved_pg: np.ndarray,
    saved_objective: np.ndarray,
    split: np.ndarray,
    security_rows: np.ndarray,
    direct_h: List[np.ndarray],
    direct_phi: List[np.ndarray],
    max_samples: int,
    balance_tol: float,
    secure_tol: float,
    active_tol: float,
    stationarity_tol: float,
    stationarity_relative_tol: float,
    complementarity_tol: float,
    pips_grad_tol: float,
    objective_relative_tol: float,
) -> Dict[str, Any]:
    """Solve selected full QPs with PYPOWER PIPS, independent of SLSQP/cuts.

    PIPS uses its own scaled gradient condition for termination.  Every
    converged candidate is therefore re-certified here with independently
    rebuilt direct-topology flows and the auditor's explicit KKT certificate.
    The initialization is a deterministic interior, balance-feasible dispatch
    and does not reuse the saved label.
    """
    if max_samples <= 0:
        return {"status": "SKIPPED", "sample_count": 0}
    imp = _pypower_imports()
    target_count = min(int(max_samples), len(pd_bus))
    candidates = [0, len(pd_bus) // 2, len(pd_bus) - 1]
    for split_id in (1, 2):
        positions = np.flatnonzero(split == split_id)
        if len(positions):
            candidates.append(int(positions[0]))
    selected = list(dict.fromkeys(candidates))[:target_count]
    remaining = target_count - len(selected)
    grid_count = remaining + 2
    while len(selected) < target_count and grid_count <= 4 * target_count + 2:
        for sample_idx in np.linspace(
            0, len(pd_bus) - 1, num=grid_count, dtype=np.int64
        )[1:-1].tolist():
            if int(sample_idx) not in selected:
                selected.append(int(sample_idx))
            if len(selected) == target_count:
                break
        grid_count += 1
    if len(selected) != target_count:
        raise RuntimeError(
            f"could only select {len(selected)} of {target_count} alternative QP samples"
        )
    hessian = diags(2.0 * static["c2"], offsets=0, format="csr")
    reports = []
    started = time.perf_counter()
    for sample_idx in selected:
        effective = pd_bus[sample_idx] + static["gs_bus"]
        demand_total = float(effective.sum())
        matrices = [csr_matrix(np.ones((1, static["ngen"]), dtype=np.float64))]
        lower = [np.asarray([demand_total])]
        upper = [np.asarray([demand_total])]
        finite = np.flatnonzero(static["fmax"] < 1e8)

        base_g = static["ptdf"][:, static["bus_of_gen"]]
        base_constant = static["flow_offset"] - static["ptdf"] @ effective
        matrices.append(csr_matrix(base_g[finite]))
        lower.append(-static["fmax"][finite] - base_constant[finite])
        upper.append(static["fmax"][finite] - base_constant[finite])
        for outage_row, hmat, phi in zip(security_rows, direct_h, direct_phi):
            keep = finite[finite != int(outage_row)]
            gmat = hmat[:, static["bus_of_gen"]]
            constant = phi - hmat @ effective
            matrices.append(csr_matrix(gmat[keep]))
            lower.append(-static["fmax"][keep] - constant[keep])
            upper.append(static["fmax"][keep] - constant[keep])
        matrix = vstack(matrices, format="csr")
        lo = np.concatenate(lower)
        hi = np.concatenate(upper)

        available = static["pmax"] - static["pmin"]
        available_total = float(np.sum(available))
        if available_total <= 0.0:
            raise ValueError("cannot construct independent PIPS initialization")
        alpha = (demand_total - float(np.sum(static["pmin"]))) / available_total
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"demand outside aggregate generation bounds for sample {sample_idx}"
            )
        independent_x0 = static["pmin"] + alpha * available
        x, _, exitflag, output, _ = imp["qps_pypower"](
            hessian,
            static["c1"],
            matrix,
            lo,
            hi,
            static["pmin"],
            static["pmax"],
            independent_x0,
            {
                "alg": 200,
                "verbose": 0,
                "max_it": 1000,
                "pips_opt": {
                    "feastol": 1e-9,
                    "gradtol": pips_grad_tol,
                    "comptol": 1e-8,
                    "costtol": 1e-10,
                },
            },
        )
        final_history = output.get("hist", [])[-1] if output.get("hist") else {}
        solver_diagnostic = {
            "exitflag": None if exitflag is None else int(exitflag),
            "message": str(output.get("message", "")),
            "iterations": int(output.get("iterations", -1)),
            "final_scaled_conditions": {
                key: float(final_history[key])
                for key in ("feascond", "gradcond", "compcond", "costcond")
                if key in final_history
            },
        }
        if exitflag is None or int(exitflag) != 1 or not np.all(np.isfinite(x)):
            raise ValueError(
                f"alternative PIPS QP failed for sample {sample_idx}: "
                f"{solver_diagnostic}"
            )
        alt_pg = np.asarray(x, dtype=np.float64)
        alt_objective = float(
            np.sum(
                static["c2"] * alt_pg * alt_pg
                + static["c1"] * alt_pg
                + static["c0"]
            )
        )
        alt_injection = -effective.copy()
        np.add.at(alt_injection, static["bus_of_gen"], alt_pg)
        base_flow = static["ptdf"] @ alt_injection + static["flow_offset"]
        worst = max(0.0, float(np.max(np.abs(base_flow) - static["fmax"])))
        contingency_flows = []
        for outage_row, hmat, phi in zip(security_rows, direct_h, direct_phi):
            flow = hmat @ alt_injection + phi
            contingency_flows.append(flow)
            violation = np.abs(flow) - static["fmax"]
            violation[int(outage_row)] = -np.inf
            worst = max(worst, 0.0, float(np.max(violation)))
        balance_abs = abs(float(np.sum(alt_pg) - demand_total))
        bound_violation = max(
            0.0,
            float(np.max(static["pmin"] - alt_pg)),
            float(np.max(alt_pg - static["pmax"])),
        )
        candidate_kkt = _kkt_certificate(
            static,
            alt_pg,
            base_flow,
            contingency_flows,
            direct_h,
            active_tol,
        )
        report = {
            "sample_index": int(sample_idx),
            **solver_diagnostic,
            "initialization": "independent_proportional_range_balance_feasible",
            "dispatch_abs_max_difference_mw": float(
                np.max(np.abs(alt_pg - saved_pg[sample_idx]))
            ),
            "objective_abs_difference": float(
                abs(alt_objective - saved_objective[sample_idx])
            ),
            "objective_relative_difference": float(
                abs(alt_objective - saved_objective[sample_idx])
                / max(1.0, abs(saved_objective[sample_idx]))
            ),
            "balance_abs_mw": balance_abs,
            "generator_bound_violation_abs_max_mw": bound_violation,
            "security_violation_abs_max_mw": worst,
            "kkt": candidate_kkt,
        }
        candidate_pass = (
            report["objective_relative_difference"] <= objective_relative_tol
            and balance_abs <= balance_tol
            and bound_violation <= balance_tol
            and worst <= secure_tol
            and candidate_kkt["stationarity_abs_max"] <= stationarity_tol
            and candidate_kkt["stationarity_relative_max"]
            <= stationarity_relative_tol
            and candidate_kkt["complementarity_abs_max"] <= complementarity_tol
        )
        if not candidate_pass:
            raise ValueError(f"alternative QP disagreement for sample {sample_idx}: {report}")
        reports.append(report)
    return {
        "status": "PASS",
        "solver": "PYPOWER PIPS primal-dual interior-point QP",
        "solver_convergence_required": True,
        "saved_label_used_as_initialization": False,
        "pips_grad_tol": pips_grad_tol,
        "objective_relative_tolerance": objective_relative_tol,
        "requested_sample_count": int(max_samples),
        "sample_count": len(reports),
        "samples": reports,
        "dispatch_abs_max_difference_mw": max(
            item["dispatch_abs_max_difference_mw"] for item in reports
        ),
        "objective_relative_difference_max": max(
            item["objective_relative_difference"] for item in reports
        ),
        "security_violation_abs_max_mw": max(
            item["security_violation_abs_max_mw"] for item in reports
        ),
        "kkt_stationarity_abs_max": max(
            item["kkt"]["stationarity_abs_max"] for item in reports
        ),
        "kkt_stationarity_relative_max": max(
            item["kkt"]["stationarity_relative_max"] for item in reports
        ),
        "kkt_complementarity_abs_max": max(
            item["kkt"]["complementarity_abs_max"] for item in reports
        ),
        "elapsed_sec": time.perf_counter() - started,
    }


def _kkt_certificate(
    static: Dict[str, Any],
    pg: np.ndarray,
    base_flow: np.ndarray,
    contingency_flows: List[np.ndarray],
    direct_h: List[np.ndarray],
    active_tol: float,
) -> Dict[str, float]:
    gradients: List[np.ndarray] = []
    slacks: List[float] = []
    base_g = static["ptdf"][:, static["bus_of_gen"]]

    def add_flow_state(flow: np.ndarray, gmat: np.ndarray) -> None:
        upper_slack = static["fmax"] - flow
        lower_slack = static["fmax"] + flow
        for row in np.flatnonzero(upper_slack <= active_tol):
            gradients.append(gmat[int(row)])
            slacks.append(float(upper_slack[int(row)]))
        for row in np.flatnonzero(lower_slack <= active_tol):
            gradients.append(-gmat[int(row)])
            slacks.append(float(lower_slack[int(row)]))

    add_flow_state(base_flow, base_g)
    for flow, hmat in zip(contingency_flows, direct_h):
        add_flow_state(flow, hmat[:, static["bus_of_gen"]])
    lower_slack = pg - static["pmin"]
    upper_slack = static["pmax"] - pg
    eye = np.eye(static["ngen"], dtype=np.float64)
    for idx in np.flatnonzero(lower_slack <= active_tol):
        gradients.append(-eye[int(idx)])
        slacks.append(float(lower_slack[int(idx)]))
    for idx in np.flatnonzero(upper_slack <= active_tol):
        gradients.append(eye[int(idx)])
        slacks.append(float(upper_slack[int(idx)]))

    grad_obj = 2.0 * static["c2"] * pg + static["c1"]
    if gradients:
        active = np.asarray(gradients, dtype=np.float64)
        matrix = np.hstack([active.T, np.ones((static["ngen"], 1))])
        lower_bounds = np.concatenate([np.zeros(len(gradients)), [-np.inf]])
    else:
        active = np.zeros((0, static["ngen"]), dtype=np.float64)
        matrix = np.ones((static["ngen"], 1), dtype=np.float64)
        lower_bounds = np.asarray([-np.inf])
    recovered = lsq_linear(
        matrix,
        -grad_obj,
        bounds=(lower_bounds, np.full(matrix.shape[1], np.inf)),
        method="bvls",
        max_iter=10000,
        tol=1e-12,
    )
    if not bool(recovered.success):
        raise RuntimeError(f"KKT multiplier recovery failed: {recovered.message}")
    multipliers = recovered.x
    residual = matrix @ multipliers + grad_obj
    inequality_mu = multipliers[: len(gradients)] if gradients else np.zeros(0)
    complementarity = (
        float(np.max(np.abs(inequality_mu * np.asarray(slacks))))
        if len(inequality_mu)
        else 0.0
    )
    return {
        "active_inequalities": int(len(gradients)),
        "stationarity_abs_max": float(np.max(np.abs(residual))),
        "stationarity_relative_max": float(
            np.max(np.abs(residual)) / max(1.0, np.max(np.abs(grad_obj)))
        ),
        "stationarity_l2": float(np.linalg.norm(residual)),
        "complementarity_abs_max": complementarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-N", type=int)
    parser.add_argument("--balance-tol", type=float, default=1e-6)
    parser.add_argument("--flow-tol", type=float, default=1e-4)
    parser.add_argument("--secure-tol", type=float, default=1e-3)
    parser.add_argument("--operator-tol", type=float, default=2e-6)
    parser.add_argument("--kkt-active-tol", type=float, default=2e-4)
    parser.add_argument("--kkt-stationarity-tol", type=float, default=2e-2)
    parser.add_argument("--kkt-stationarity-relative-tol", type=float, default=2e-5)
    parser.add_argument("--kkt-complementarity-tol", type=float, default=1e-5)
    parser.add_argument("--alternative-solver-samples", type=int, default=5)
    parser.add_argument("--alternative-pips-grad-tol", type=float, default=1e-4)
    parser.add_argument(
        "--alternative-objective-relative-tol", type=float, default=1e-10
    )
    parser.add_argument("--direct-progress-every", type=int, default=8)
    parser.add_argument("--provenance-out", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("dataset_schema_name") != SCHEMA_NAME:
        raise ValueError("unexpected preventive SC-DCOPF schema name")
    if int(meta.get("dataset_schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unexpected preventive SC-DCOPF schema version")
    if meta.get("case_name") != CASE_NAME:
        raise ValueError("audit is restricted to IEEE 300")

    manifest_path = args.manifest or args.data.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema") != "phase_a_scopf_generation_manifest_v2":
        raise ValueError("unexpected generation manifest schema")
    if sha256_file(args.data) != manifest["dataset"]["sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    if sha256_file(args.meta) != manifest["metadata"]["sha256"]:
        raise ValueError("metadata SHA-256 mismatch")
    for name, entry in manifest.get("source", {}).items():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"generation source hash mismatch: {name}")

    with np.load(args.data, allow_pickle=False) as z:
        required = {
            "pd_bus", "pg_scopf", "base_flow", "objective", "split", "source_index",
            "gs_bus", "ptdf", "flow_offset", "fmax", "pmin", "pmax", "cost_c2",
            "cost_c1", "cost_c0", "bus_of_gen", "security_branch_rows",
            "security_line_pp_indices", "security_lodf_columns", "pd_bus_includes_gs",
            "load_scale_factors", "worst_security_violation_mw",
            "worst_contingency_position", "worst_branch_row", "cut_rounds",
            "active_security_rows", "qp_iterations", "label_time_sec",
            "solver_success", "solver_status", "solver_message", "balance_abs_mw",
            "generator_bound_violation_mw", "base_flow_violation_mw",
            "diagnostic_tie_tolerance_mw",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"missing dataset fields: {missing}")
        saved = {name: z[name] for name in z.files}
    pd_bus = saved["pd_bus"].astype(np.float64)
    pg = saved["pg_scopf"].astype(np.float64)
    saved_base_flow = saved["base_flow"].astype(np.float64)
    objective = saved["objective"].astype(np.float64)
    split = saved["split"].astype(np.int64)
    source_index = saved["source_index"].astype(np.int64)
    security_rows = saved["security_branch_rows"].astype(np.int64)
    security_pp = saved["security_line_pp_indices"].astype(np.int64)
    saved_lodf = saved["security_lodf_columns"].astype(np.float64)
    diagnostic_tie_tol = float(
        np.asarray(saved["diagnostic_tie_tolerance_mw"]).item()
    )
    if not np.isclose(
        diagnostic_tie_tol,
        float(meta["label_solver"]["diagnostic_identity_tie_tolerance_mw"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("diagnostic tie tolerance disagrees between data and metadata")
    n = pd_bus.shape[0]
    if args.expected_N is not None and n != args.expected_N:
        raise ValueError(f"dataset N={n}, expected {args.expected_N}")
    if int(np.asarray(saved["pd_bus_includes_gs"]).item()) != 0:
        raise ValueError("pd_bus_includes_gs must be zero")
    if pg.shape[0] != n or saved_base_flow.shape[0] != n:
        raise ValueError("sample array counts disagree")
    if len(np.unique(source_index)) != n:
        raise ValueError("source_index is not unique")
    per_sample_fields = [
        "objective", "split", "source_index", "worst_security_violation_mw",
        "worst_contingency_position", "worst_branch_row", "cut_rounds",
        "active_security_rows", "qp_iterations", "label_time_sec",
        "solver_success", "solver_status", "solver_message", "balance_abs_mw",
        "generator_bound_violation_mw", "base_flow_violation_mw",
    ]
    for name in per_sample_fields:
        if np.asarray(saved[name]).shape != (n,):
            raise ValueError(f"per-sample field {name} has wrong shape")
    for name, value in saved.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"numeric dataset field {name} contains nonfinite values")
    solver_messages = saved["solver_message"].astype(str)
    solver_evidence_ok = (
        np.all(saved["solver_success"].astype(np.int64) == 1)
        and np.all(saved["solver_status"].astype(np.int64) == 0)
        and all(bool(message.strip()) for message in solver_messages.tolist())
        and np.all(saved["qp_iterations"].astype(np.int64) > 0)
        and np.all(saved["cut_rounds"].astype(np.int64) >= 0)
        and np.all(saved["active_security_rows"].astype(np.int64) >= 0)
        and np.all(saved["label_time_sec"].astype(np.float64) > 0.0)
    )

    net, ppc, static = build_network_contract()
    full_security, expected_security = _independent_security_sets(
        static, declared_count=int(meta["declared_contingencies"])
    )
    if not np.array_equal(security_rows, expected_security["branch_rows"]):
        raise ValueError("saved security branch rows do not match deterministic selection")
    if not np.array_equal(security_pp, expected_security["line_pp_indices"]):
        raise ValueError("saved pandapower line indices do not match deterministic selection")
    payload = [
        {"line_pp_idx": int(pp), "branch_row": int(row)}
        for pp, row in zip(security_pp, security_rows)
    ]
    if stable_json_digest(payload) != meta["contingency_set_digest"]:
        raise ValueError("contingency set digest mismatch")
    if expected_security["digest"] != meta["contingency_set_digest"]:
        raise ValueError("independent severity selection digest mismatch")
    lodf_saved_selection_error = float(
        np.max(np.abs(saved_lodf - expected_security["lodf_columns"]))
    )
    if lodf_saved_selection_error > args.operator_tol:
        raise ValueError("saved LODFs disagree with independent eligibility derivation")
    graph_bridges = _graph_bridge_rows(static)
    independently_excluded = np.asarray(
        [item["branch_row"] for item in full_security["excluded"]], dtype=np.int64
    )
    if not np.array_equal(np.sort(graph_bridges), np.sort(independently_excluded)):
        raise ValueError("graph bridges and singular LODF outage sets disagree")

    array_pairs = {
        "gs_bus": static["gs_bus"],
        "ptdf": static["ptdf"],
        "flow_offset": static["flow_offset"],
        "fmax": static["fmax"],
        "pmin": static["pmin"],
        "pmax": static["pmax"],
        "cost_c2": static["c2"],
        "cost_c1": static["c1"],
        "cost_c0": static["c0"],
        "bus_of_gen": static["bus_of_gen"],
    }
    static_errors = {}
    for name, rebuilt in array_pairs.items():
        stored = np.asarray(saved[name])
        err = float(np.max(np.abs(stored.astype(np.float64) - np.asarray(rebuilt, dtype=np.float64))))
        static_errors[name] = err
        if err > 1e-10:
            raise ValueError(f"saved static array {name} differs from rebuilt network: {err}")
    rebuilt_hashes = static_hashes(static, expected_security)
    if rebuilt_hashes != meta["static_hashes"]:
        raise ValueError("static SHA-256 contract mismatch")

    unique_split = set(np.unique(split).tolist())
    if n >= 3 and unique_split != {0, 1, 2}:
        raise ValueError(f"expected split IDs 0,1,2; got {sorted(unique_split)}")
    digests = [[_row_digest(row) for row in pd_bus[split == sid]] for sid in (0, 1, 2)]
    digest_sets = [set(rows) for rows in digests]
    if digest_sets[0] & digest_sets[1] or digest_sets[0] & digest_sets[2] or digest_sets[1] & digest_sets[2]:
        raise ValueError("exact demand rows overlap across splits")
    all_digests = [_row_digest(row) for row in pd_bus]
    exact_demand_duplicates_all = n - len(set(all_digests))
    if exact_demand_duplicates_all:
        raise ValueError(f"dataset contains {exact_demand_duplicates_all} duplicate demands")

    factors = saved["load_scale_factors"].astype(np.float64)
    if factors.shape != (n, len(static["load_base_p_mw"])):
        raise ValueError("load_scale_factors shape does not match pandapower loads")
    scale_low, scale_high = map(float, meta["demand_scale_range"])
    factor_range_violation = max(
        0.0,
        float(scale_low - np.min(factors)),
        float(np.max(factors) - scale_high),
    )
    reconstructed_pd = np.repeat(static["base_pd"][None, :], n, axis=0)
    for sample_idx in range(n):
        delta = static["load_base_p_mw"] * (factors[sample_idx] - 1.0)
        np.add.at(reconstructed_pd[sample_idx], static["load_internal_buses"], delta)
    demand_reconstruction_error = float(np.max(np.abs(reconstructed_pd - pd_bus)))
    if factor_range_violation > 1e-12 or demand_reconstruction_error > 1e-9:
        raise ValueError(
            f"demand sampling contract failed: factor={factor_range_violation}, "
            f"reconstruction={demand_reconstruction_error}"
        )

    full_set_certificate = _independent_full_set_feasibility(static, full_security)
    generator_full_set_certificate = meta.get(
        "full_eligible_set_base_demand_certificate", {}
    )
    if (
        full_set_certificate["status"] != "INFEASIBLE"
        or generator_full_set_certificate.get("status") != "INFEASIBLE"
    ):
        raise ValueError(
            "full eligible preventive set was not independently certified infeasible"
        )

    direct_h, direct_phi, topology_timing = _direct_outage_operators(
        ppc, static, security_rows, args.direct_progress_every
    )
    lodf_h_error = 0.0
    lodf_phi_error = 0.0
    independent_lodf_columns = []
    for pos, (outage_row, h_direct, phi_direct) in enumerate(
        zip(security_rows, direct_h, direct_phi)
    ):
        col = saved_lodf[:, pos]
        h_lodf = static["ptdf"] + col[:, None] * static["ptdf"][int(outage_row)][None, :]
        phi_lodf = static["flow_offset"] + col * static["flow_offset"][int(outage_row)]
        lodf_h_error = max(lodf_h_error, float(np.max(np.abs(h_direct - h_lodf))))
        lodf_phi_error = max(lodf_phi_error, float(np.max(np.abs(phi_direct - phi_lodf))))
        # Independently derive the column from a nonzero entry of the outaged
        # base PTDF row, then enforce the standard -1 diagonal convention.
        probe_col = int(np.argmax(np.abs(static["ptdf"][int(outage_row)])))
        denom = float(static["ptdf"][int(outage_row), probe_col])
        if abs(denom) < 1e-12:
            raise ValueError("cannot derive independent LODF column")
        independent = (h_direct[:, probe_col] - static["ptdf"][:, probe_col]) / denom
        independent[int(outage_row)] = -1.0
        independent_lodf_columns.append(independent)
    independent_lodf = np.stack(independent_lodf_columns, axis=1)
    lodf_column_error = float(np.max(np.abs(independent_lodf - saved_lodf)))
    if max(lodf_h_error, lodf_phi_error, lodf_column_error) > args.operator_tol:
        raise ValueError(
            f"direct topology versus LODF operator mismatch: H={lodf_h_error}, "
            f"phi={lodf_phi_error}, LODF={lodf_column_error}"
        )

    effective_demand = pd_bus + static["gs_bus"][None, :]
    injection = -effective_demand.copy()
    for gen_idx, bus_idx in enumerate(static["bus_of_gen"]):
        injection[:, int(bus_idx)] += pg[:, gen_idx]
    balance = injection.sum(axis=1)
    bound_violation = np.maximum(static["pmin"][None, :] - pg, 0.0)
    bound_violation = np.maximum(bound_violation, pg - static["pmax"][None, :])
    base_flow = injection @ static["ptdf"].T + static["flow_offset"][None, :]
    base_flow_saved_error = float(np.max(np.abs(base_flow - saved_base_flow)))
    objective_rebuilt = np.sum(
        static["c2"][None, :] * pg * pg
        + static["c1"][None, :] * pg
        + static["c0"][None, :],
        axis=1,
    )
    objective_error = float(np.max(np.abs(objective_rebuilt - objective)))
    base_raw_violation = np.abs(base_flow) - static["fmax"][None, :]
    base_violation = np.maximum(base_raw_violation, 0.0)

    per_sample_raw_worst = np.max(base_raw_violation, axis=1)
    per_sample_worst_cont = np.full(n, -1, dtype=np.int64)
    per_sample_worst_branch = np.argmax(base_raw_violation, axis=1).astype(np.int64)
    outaged_flow_abs_max = 0.0
    direct_vs_lodf_flow_error = 0.0
    contingency_flows_by_sample: List[List[np.ndarray]] = [[] for _ in range(n)]
    for pos, (outage_row, h_direct, phi_direct) in enumerate(
        zip(security_rows, direct_h, direct_phi)
    ):
        direct_flow = injection @ h_direct.T + phi_direct[None, :]
        lodf_flow = base_flow + independent_lodf[:, pos][None, :] * base_flow[:, int(outage_row), None]
        direct_vs_lodf_flow_error = max(
            direct_vs_lodf_flow_error, float(np.max(np.abs(direct_flow - lodf_flow)))
        )
        outaged_flow_abs_max = max(
            outaged_flow_abs_max, float(np.max(np.abs(direct_flow[:, int(outage_row)])))
        )
        violation = np.abs(direct_flow) - static["fmax"][None, :]
        violation[:, int(outage_row)] = -np.inf
        state_worst = np.max(violation, axis=1)
        state_worst_branch = np.argmax(violation, axis=1)
        better = state_worst > per_sample_raw_worst
        per_sample_raw_worst[better] = state_worst[better]
        per_sample_worst_cont[better] = pos
        per_sample_worst_branch[better] = state_worst_branch[better]
        for sample_idx in range(n):
            contingency_flows_by_sample[sample_idx].append(direct_flow[sample_idx])
    per_sample_worst = np.maximum(per_sample_raw_worst, 0.0)
    # Canonicalize identities independently. Multiple active constraints are
    # often tied within roundoff, so exact argmax identity is not portable
    # between algebraically equivalent direct-PTDF and LODF evaluations.
    for sample_idx in range(n):
        threshold = per_sample_raw_worst[sample_idx] - diagnostic_tie_tol
        tied_base = np.flatnonzero(base_raw_violation[sample_idx] >= threshold)
        if len(tied_base):
            per_sample_worst_cont[sample_idx] = -1
            per_sample_worst_branch[sample_idx] = int(tied_base[0])
            continue
        found = False
        for cont_pos, (outage_row, flow) in enumerate(
            zip(security_rows, contingency_flows_by_sample[sample_idx])
        ):
            raw = np.abs(flow) - static["fmax"]
            raw[int(outage_row)] = -np.inf
            tied = np.flatnonzero(raw >= threshold)
            if len(tied):
                per_sample_worst_cont[sample_idx] = cont_pos
                per_sample_worst_branch[sample_idx] = int(tied[0])
                found = True
                break
        if not found:
            raise ValueError("failed to reproduce canonical worst-constraint identity")
    saved_worst_error = float(
        np.max(
            np.abs(
                saved["worst_security_violation_mw"].astype(np.float64)
                - per_sample_worst
            )
        )
    )
    saved_worst_contingency_mismatches = int(
        np.count_nonzero(
            saved["worst_contingency_position"].astype(np.int64)
            != per_sample_worst_cont
        )
    )
    saved_worst_branch_mismatches = int(
        np.count_nonzero(
            saved["worst_branch_row"].astype(np.int64)
            != per_sample_worst_branch
        )
    )
    per_sample_bound_violation = np.max(bound_violation, axis=1)
    per_sample_base_violation = np.max(base_violation, axis=1)
    saved_balance_error = float(
        np.max(np.abs(saved["balance_abs_mw"] - np.abs(balance)))
    )
    saved_bound_error = float(
        np.max(
            np.abs(
                saved["generator_bound_violation_mw"] - per_sample_bound_violation
            )
        )
    )
    saved_base_violation_error = float(
        np.max(
            np.abs(saved["base_flow_violation_mw"] - per_sample_base_violation)
        )
    )

    # Reconstruct the same stacked operator in PyTorch float64 without using
    # any completed-study scenario-conditioned operator.
    with torch.no_grad():
        t_pg = torch.as_tensor(pg, dtype=torch.float64)
        t_pd = torch.as_tensor(pd_bus, dtype=torch.float64)
        t_ptdf = torch.as_tensor(static["ptdf"], dtype=torch.float64)
        t_gs = torch.as_tensor(static["gs_bus"], dtype=torch.float64)
        t_offset = torch.as_tensor(static["flow_offset"], dtype=torch.float64)
        t_bus = torch.as_tensor(static["bus_of_gen"], dtype=torch.long)
        t_inj = -(t_pd + t_gs[None, :])
        t_inj = t_inj.scatter_add(1, t_bus[None, :].expand(n, -1), t_pg)
        t_base = t_inj @ t_ptdf.T + t_offset[None, :]
        t_lodf = torch.as_tensor(independent_lodf.T, dtype=torch.float64)
        t_rows = torch.as_tensor(security_rows, dtype=torch.long)
        t_cont = t_base[:, None, :] + t_lodf[None, :, :] * t_base[:, t_rows, None]
        torch_base_error = float(torch.max(torch.abs(t_base - torch.as_tensor(base_flow))).item())
        torch_cont_error = 0.0
        for pos in range(len(security_rows)):
            direct_tensor = torch.as_tensor(
                np.stack([contingency_flows_by_sample[i][pos] for i in range(n)]),
                dtype=torch.float64,
            )
            torch_cont_error = max(
                torch_cont_error,
                float(torch.max(torch.abs(t_cont[:, pos, :] - direct_tensor)).item()),
            )

    kkt_reports = []
    for sample_idx in range(n):
        kkt_reports.append(
            _kkt_certificate(
                static,
                pg[sample_idx],
                base_flow[sample_idx],
                contingency_flows_by_sample[sample_idx],
                direct_h,
                args.kkt_active_tol,
            )
        )
    kkt_stationarity_max = max(x["stationarity_abs_max"] for x in kkt_reports)
    kkt_stationarity_relative_max = max(
        x["stationarity_relative_max"] for x in kkt_reports
    )
    kkt_complementarity_max = max(x["complementarity_abs_max"] for x in kkt_reports)
    alternative_solver = _alternative_qp_crosscheck(
        static,
        pd_bus,
        pg,
        objective,
        split,
        security_rows,
        direct_h,
        direct_phi,
        args.alternative_solver_samples,
        args.balance_tol,
        args.secure_tol,
        args.kkt_active_tol,
        args.kkt_stationarity_tol,
        args.kkt_stationarity_relative_tol,
        args.kkt_complementarity_tol,
        args.alternative_pips_grad_tol,
        args.alternative_objective_relative_tol,
    )

    gates = {
        "finite_numeric_arrays": True,
        "solver_evidence": bool(solver_evidence_ok),
        "unique_demands_all_splits": exact_demand_duplicates_all == 0,
        "demand_reconstruction": demand_reconstruction_error <= 1e-9,
        "graph_bridge_outage_census": np.array_equal(
            np.sort(graph_bridges), np.sort(independently_excluded)
        ),
        "full_eligible_set_infeasible": (
            full_set_certificate["status"] == "INFEASIBLE"
            and generator_full_set_certificate.get("status") == "INFEASIBLE"
        ),
        "balance": float(np.max(np.abs(balance))) <= args.balance_tol,
        "generator_bounds": float(np.max(bound_violation)) <= args.balance_tol,
        "base_flow": float(np.max(base_violation)) <= args.flow_tol,
        "all_contingencies": float(np.max(per_sample_worst)) <= args.flow_tol,
        "all_samples_secure_1e3": float(np.max(per_sample_worst)) <= args.secure_tol,
        "outaged_line_zero": outaged_flow_abs_max <= args.operator_tol,
        "saved_base_flow": base_flow_saved_error <= args.operator_tol,
        "objective": objective_error <= 1e-6,
        "saved_label_diagnostics": (
            saved_worst_error <= args.operator_tol
            and saved_worst_contingency_mismatches == 0
            and saved_worst_branch_mismatches == 0
            and max(saved_balance_error, saved_bound_error, saved_base_violation_error)
            <= args.operator_tol
        ),
        "direct_topology_operator": max(lodf_h_error, lodf_phi_error, lodf_column_error) <= args.operator_tol,
        "torch_operator": max(torch_base_error, torch_cont_error) <= args.operator_tol,
        "kkt_stationarity": (
            kkt_stationarity_max <= args.kkt_stationarity_tol
            and kkt_stationarity_relative_max
            <= args.kkt_stationarity_relative_tol
        ),
        "kkt_complementarity": (
            kkt_complementarity_max <= args.kkt_complementarity_tol
        ),
        "alternative_qp_solver": (
            alternative_solver["status"] == "PASS"
            or (
                args.alternative_solver_samples <= 0
                and alternative_solver["status"] == "SKIPPED"
            )
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "failed_gates": failed,
                    "saved_label_diagnostic_debug": {
                        "worst_value_abs_max_error_mw": saved_worst_error,
                        "worst_contingency_mismatches": saved_worst_contingency_mismatches,
                        "worst_branch_mismatches": saved_worst_branch_mismatches,
                        "balance_abs_max_error_mw": saved_balance_error,
                        "bound_abs_max_error_mw": saved_bound_error,
                        "base_violation_abs_max_error_mw": saved_base_violation_error,
                    },
                },
                indent=2,
            ),
            flush=True,
        )
        raise ValueError(f"audit gates failed: {failed}")

    provenance_out = args.provenance_out or args.data.with_suffix(".provenance.json")
    report = {
        "status": "PASS",
        "case": CASE_NAME,
        "N": n,
        "dimensions": {
            "nbus": static["nbus"],
            "ngen": static["ngen"],
            "nbranch": static["nbranch"],
        },
        "declared_contingencies": int(len(security_rows)),
        "eligible_non_islanding_contingencies": int(len(full_security["branch_rows"])),
        "excluded_islanding_or_singular_contingencies": int(len(full_security["excluded"])),
        "security_set_digest": meta["contingency_set_digest"],
        "split_counts": {
            "train": int(np.count_nonzero(split == 0)),
            "validation": int(np.count_nonzero(split == 1)),
            "test": int(np.count_nonzero(split == 2)),
        },
        "exact_demand_cross_split_duplicates": 0,
        "exact_demand_duplicates_all": exact_demand_duplicates_all,
        "audit_independence": {
            "label_solver_called": False,
            "contingency_flow_helper_called": False,
            "eligibility_lodf_and_selection_rederived_in_auditor": True,
            "declared_outage_topologies_rebuilt_directly": True,
            "alternative_optimizer": "PYPOWER PIPS",
        },
        "gates": gates,
        "metrics": {
            "balance_abs_max_mw": float(np.max(np.abs(balance))),
            "generator_bound_violation_abs_max_mw": float(np.max(bound_violation)),
            "base_flow_violation_abs_max_mw": float(np.max(base_violation)),
            "all_contingency_violation_abs_max_mw": float(np.max(per_sample_worst)),
            "all_contingency_violation_mean_mw": float(np.mean(per_sample_worst)),
            "all_contingency_secure_rate_at_1e3_mw": float(np.mean(per_sample_worst <= args.secure_tol)),
            "outaged_line_flow_abs_max_mw": outaged_flow_abs_max,
            "saved_base_flow_reconstruction_abs_max_mw": base_flow_saved_error,
            "objective_reconstruction_abs_max": objective_error,
            "demand_reconstruction_abs_max_mw": demand_reconstruction_error,
            "load_factor_range_violation_abs_max": factor_range_violation,
            "saved_worst_security_abs_max_error_mw": saved_worst_error,
            "saved_worst_contingency_mismatches": saved_worst_contingency_mismatches,
            "saved_worst_branch_mismatches": saved_worst_branch_mismatches,
            "saved_balance_diagnostic_abs_max_error_mw": saved_balance_error,
            "saved_bound_diagnostic_abs_max_error_mw": saved_bound_error,
            "saved_base_violation_diagnostic_abs_max_error_mw": saved_base_violation_error,
            "independent_selection_lodf_abs_max_error": lodf_saved_selection_error,
            "direct_vs_lodf_H_abs_max": lodf_h_error,
            "direct_vs_lodf_phase_offset_abs_max_mw": lodf_phi_error,
            "independent_lodf_column_abs_max": lodf_column_error,
            "direct_vs_lodf_sample_flow_abs_max_mw": direct_vs_lodf_flow_error,
            "torch_base_operator_abs_max_mw": torch_base_error,
            "torch_contingency_operator_abs_max_mw": torch_cont_error,
            "kkt_stationarity_abs_max": kkt_stationarity_max,
            "kkt_stationarity_relative_max": kkt_stationarity_relative_max,
            "kkt_complementarity_abs_max": kkt_complementarity_max,
            "kkt_active_inequalities_max": int(max(x["active_inequalities"] for x in kkt_reports)),
        },
        "static_rebuild_abs_max": static_errors,
        "topology_validation": {
            "direct_post_outage_operators_built": int(len(direct_h)),
            "graph_bridge_count": int(len(graph_bridges)),
            "singular_lodf_denominator_count": int(len(independently_excluded)),
            **topology_timing,
        },
        "solver_evidence": {
            "all_success": bool(np.all(saved["solver_success"] == 1)),
            "status_values": sorted(
                int(value)
                for value in np.unique(saved["solver_status"]).tolist()
            ),
            "message_values": sorted(set(solver_messages.tolist())),
            "qp_iterations_min": int(np.min(saved["qp_iterations"])),
            "qp_iterations_max": int(np.max(saved["qp_iterations"])),
            "cut_rounds_min": int(np.min(saved["cut_rounds"])),
            "cut_rounds_max": int(np.max(saved["cut_rounds"])),
            "label_time_sec_min": float(np.min(saved["label_time_sec"])),
            "label_time_sec_max": float(np.max(saved["label_time_sec"])),
        },
        "full_eligible_set_validation": {
            "generator_certificate": generator_full_set_certificate,
            "independent_auditor_certificate": full_set_certificate,
        },
        "alternative_solver_validation": alternative_solver,
        "provenance": {
            "dataset_sha256": sha256_file(args.data),
            "metadata_sha256": sha256_file(args.meta),
            "manifest_sha256": sha256_file(manifest_path),
            "final_provenance_path": str(provenance_out),
        },
        "audit_wall_time_sec": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    source_root = Path(__file__).resolve().parent
    source_paths = {
        "generator": source_root / "generate_phase_a_smoke.py",
        "auditor": Path(__file__).resolve(),
        "common": source_root / "phase_a_scopf" / "common.py",
        "package_init": source_root / "phase_a_scopf" / "__init__.py",
        "deterministic_tests": source_root / "test_phase_a_scopf.py",
    }
    final_provenance = {
        "manifest_schema": "phase_a_scopf_audited_provenance_v2",
        "status": "AUDITED_PASS",
        "artifacts": {
            "dataset": {
                "path": str(args.data.resolve()),
                "sha256": sha256_file(args.data),
            },
            "metadata": {
                "path": str(args.meta.resolve()),
                "sha256": sha256_file(args.meta),
            },
            "generation_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "audit_report": {
                "path": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
            },
        },
        "source": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "configuration": {
            "case": CASE_NAME,
            "N": n,
            "declared_contingencies": int(len(security_rows)),
            "security_set_digest": meta["contingency_set_digest"],
            "audit_tolerances": {
                "balance_mw": args.balance_tol,
                "flow_mw": args.flow_tol,
                "secure_mw": args.secure_tol,
                "operator": args.operator_tol,
                "kkt_stationarity_abs": args.kkt_stationarity_tol,
                "kkt_stationarity_relative": args.kkt_stationarity_relative_tol,
                "kkt_complementarity": args.kkt_complementarity_tol,
                "alternative_pips_gradient_scaled": args.alternative_pips_grad_tol,
                "alternative_objective_relative": args.alternative_objective_relative_tol,
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "torch": torch.__version__,
            "pandapower": importlib.metadata.version("pandapower"),
            "pypower": importlib.metadata.version("PYPOWER"),
        },
    }
    provenance_out.parent.mkdir(parents=True, exist_ok=True)
    provenance_out.write_text(json.dumps(final_provenance, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
