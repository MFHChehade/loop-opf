#!/usr/bin/env python3
"""Deterministic mathematical smoke checks for preventive IEEE 300 SC-DCOPF."""

from __future__ import annotations

import argparse
import json

import numpy as np

from phase_a_scopf.common import (
    all_contingency_flows,
    build_network_contract,
    discover_line_security_set,
    objective_value,
    select_smoke_security_set,
    security_violation_matrix,
    solve_preventive_scopf,
)


def run() -> dict:
    _, _, static = build_network_contract()
    full_security = discover_line_security_set(static)
    expected = {"nbus": 300, "ngen": 69, "nbranch": 411}
    actual = {key: int(static[key]) for key in expected}
    if actual != expected:
        raise AssertionError(f"unexpected IEEE 300 dimensions: {actual}")
    if len(static["line_branch_rows"]) != 283:
        raise AssertionError("unexpected pandapower line count")
    if len(full_security["branch_rows"]) != 253 or len(full_security["excluded"]) != 30:
        raise AssertionError("unexpected valid/islanding line-contingency census")
    full_set_status = "unexpectedly_feasible"
    full_set_reason = ""
    try:
        solve_preventive_scopf(static, full_security, static["base_pd"])
    except RuntimeError as exc:
        full_set_reason = str(exc)
        if "linearly infeasible" not in full_set_reason:
            raise
        full_set_status = "INFEASIBLE"
    if full_set_status != "INFEASIBLE":
        raise AssertionError("expected inherited-limit full eligible set infeasibility")
    security = select_smoke_security_set(static, full_security, count=32)
    diag = security["lodf_columns"][
        security["branch_rows"], np.arange(len(security["branch_rows"]))
    ]
    if not np.allclose(diag, -1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("LODF diagonal contract failed")

    solved = solve_preventive_scopf(static, security, static["base_pd"])
    if not solved.solver_success or solved.solver_status != 0:
        raise AssertionError(
            f"label solver did not terminate optimally: {solved.solver_status} "
            f"{solved.solver_message}"
        )
    if not solved.solver_message.strip():
        raise AssertionError("label solver did not provide a termination message")
    effective = static["base_pd"] + static["gs_bus"]
    balance = float(solved.pg.sum() - effective.sum())
    if abs(balance) > 1e-6:
        raise AssertionError(f"balance failed: {balance} MW")
    if abs(solved.balance_abs_mw - abs(balance)) > 1e-10:
        raise AssertionError("saved balance diagnostic is inconsistent")
    if np.max(static["pmin"] - solved.pg) > 1e-6:
        raise AssertionError("generator lower bound failed")
    if np.max(solved.pg - static["pmax"]) > 1e-6:
        raise AssertionError("generator upper bound failed")
    flows = all_contingency_flows(solved.base_flow, security)
    outaged = np.abs(
        flows[np.arange(len(security["branch_rows"])), security["branch_rows"]]
    )
    if np.max(outaged) > 1e-8:
        raise AssertionError("outaged-line zero-flow contract failed")
    violation = security_violation_matrix(
        flows, static["fmax"], security["branch_rows"]
    )
    all_worst = float(max(0.0, np.max(violation)))
    if all_worst > 1e-4:
        raise AssertionError(f"all-contingency security failed: {all_worst} MW")
    recomputed_objective = objective_value(static, solved.pg)
    if not np.isclose(recomputed_objective, solved.objective, atol=1e-8, rtol=1e-12):
        raise AssertionError("objective recomputation failed")
    return {
        "status": "PASS",
        "dimensions": actual,
        "pandapower_lines": int(len(static["line_branch_rows"])),
        "declared_non_islanding_outages": int(len(security["branch_rows"])),
        "eligible_non_islanding_outages": int(len(full_security["branch_rows"])),
        "excluded_islanding_or_singular_outages": int(len(full_security["excluded"])),
        "screened_out_eligible_outages": int(len(security["screened_out_eligible"])),
        "selection_policy": security["selection_policy"],
        "full_eligible_set_base_demand_status": full_set_status,
        "full_eligible_set_failure": full_set_reason,
        "security_set_digest": security["digest"],
        "balance_abs_mw": abs(balance),
        "base_flow_violation_mw": float(
            max(0.0, np.max(np.abs(solved.base_flow) - static["fmax"]))
        ),
        "all_contingency_violation_mw": all_worst,
        "outaged_line_flow_abs_max_mw": float(np.max(outaged)),
        "objective": solved.objective,
        "cut_rounds": solved.cut_rounds,
        "active_security_rows": solved.active_security_rows,
        "qp_iterations": solved.qp_iterations,
        "solve_time_sec": solved.solve_time_sec,
        "solver_success": solved.solver_success,
        "solver_status": solved.solver_status,
        "solver_message": solved.solver_message,
        "saved_balance_abs_mw": solved.balance_abs_mw,
        "saved_generator_bound_violation_mw": solved.generator_bound_violation_mw,
        "saved_base_flow_violation_mw": solved.base_flow_violation_mw,
    }


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
