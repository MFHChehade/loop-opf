"""Shared network contract and preventive SC-DCOPF label solver.

The network construction intentionally matches the completed Fair4Net study,
but every label produced here solves one preventive dispatch against the full
declared line-contingency set simultaneously.  It does not reuse any
scenario-conditioned N-1 labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.sparse import csr_matrix, diags, vstack


CASE_NAME = "case300"
LINE_LIMIT_SCALE = 0.3
LOAD_SCALE = 1.0
GEN_MARGIN_MW = 5.0
VM_BOUNDS = (0.9, 1.1)
NETWORK_SEED = 7
SCHEMA_NAME = "preventive_line_sc_dc_opf"
SCHEMA_VERSION = 2
FLOW_MODEL_CONTRACT = "f=PTDF@(Pg-PD-GS)+phase_shift_offset"
SECURITY_MODEL_CONTRACT = (
    "one preventive Pg satisfies base topology and every declared "
    "non-islanding pandapower-line outage simultaneously"
)
DIAGNOSTIC_TIE_TOL_MW = 1e-7


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _pypower_imports():
    try:
        from pypower.ext2int import ext2int
        from pypower.idx_brch import BR_STATUS, F_BUS, RATE_A, SHIFT, T_BUS
        from pypower.idx_bus import BUS_TYPE, GS, PD, REF
        from pypower.idx_cost import POLYNOMIAL
        from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN
        from pypower.makeBdc import makeBdc
        from pypower.makePTDF import makePTDF
        from pypower.qps_pypower import qps_pypower
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, RATE_A, SHIFT, T_BUS
        from pandapower.pypower.idx_bus import BUS_TYPE, GS, PD, REF
        from pandapower.pypower.idx_cost import POLYNOMIAL
        from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN
        from pandapower.pypower.makeBdc import makeBdc
        from pandapower.pypower.makePTDF import makePTDF
        from pypower.qps_pypower import qps_pypower
    return {
        "ext2int": ext2int,
        "BR_STATUS": BR_STATUS,
        "F_BUS": F_BUS,
        "RATE_A": RATE_A,
        "SHIFT": SHIFT,
        "T_BUS": T_BUS,
        "BUS_TYPE": BUS_TYPE,
        "GS": GS,
        "PD": PD,
        "REF": REF,
        "POLYNOMIAL": POLYNOMIAL,
        "GEN_BUS": GEN_BUS,
        "GEN_STATUS": GEN_STATUS,
        "PG": PG,
        "PMAX": PMAX,
        "PMIN": PMIN,
        "makeBdc": makeBdc,
        "makePTDF": makePTDF,
        "qps_pypower": qps_pypower,
    }


def _dense_1d(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _flow_offset(
    make_bdc,
    base_mva: float,
    bus: np.ndarray,
    branch: np.ndarray,
    ptdf: np.ndarray,
) -> np.ndarray:
    try:
        built = make_bdc(base_mva, bus, branch)
    except TypeError:
        built = make_bdc(bus, branch)
    if len(built) < 4:
        raise RuntimeError("makeBdc did not return phase-shift injection vectors")
    pbusinj = _dense_1d(built[2])
    pfinj = _dense_1d(built[3])
    result = base_mva * (pfinj - ptdf @ pbusinj)
    if result.shape != (branch.shape[0],) or not np.all(np.isfinite(result)):
        raise RuntimeError("invalid phase-shift flow offset")
    return result


def build_base_net():
    """Build the exact case300 network contract used by the completed study."""
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case300()
    if abs(LOAD_SCALE - 1.0) > 1e-12 and len(net.load):
        net.load.loc[:, "p_mw"] *= LOAD_SCALE
        if "q_mvar" in net.load.columns:
            net.load.loc[:, "q_mvar"] *= LOAD_SCALE
    if len(net.gen):
        net.gen.loc[:, "controllable"] = True
        if "vm_pu" in net.gen.columns:
            net.gen.loc[:, "vm_pu"] = 1.0
    if len(net.ext_grid):
        net.ext_grid.loc[:, "controllable"] = True
        if "vm_pu" in net.ext_grid.columns:
            net.ext_grid.loc[:, "vm_pu"] = 1.0
    if "min_vm_pu" in net.bus.columns:
        net.bus.loc[:, "min_vm_pu"] = VM_BOUNDS[0]
    if "max_vm_pu" in net.bus.columns:
        net.bus.loc[:, "max_vm_pu"] = VM_BOUNDS[1]
    if len(net.line):
        if "max_i_ka" in net.line.columns:
            finite = np.isfinite(net.line["max_i_ka"].astype(float).to_numpy())
            net.line.loc[finite, "max_i_ka"] = (
                net.line.loc[finite, "max_i_ka"].astype(float) * LINE_LIMIT_SCALE
            )
        if "max_loading_percent" in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = (
                net.line["max_loading_percent"].astype(float) * LINE_LIMIT_SCALE
            )
    if len(net.trafo) and "max_loading_percent" in net.trafo.columns:
        net.trafo.loc[:, "max_loading_percent"] = (
            net.trafo["max_loading_percent"].astype(float) * LINE_LIMIT_SCALE
        )

    if hasattr(net, "poly_cost") and len(net.poly_cost):
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    if hasattr(net, "pwl_cost") and len(net.pwl_cost):
        net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

    elements = [("ext_grid", int(i)) for i in net.ext_grid.index]
    elements += [("gen", int(i)) for i in net.gen.index]
    if not elements:
        raise RuntimeError("case300 has no controllable generators")
    total_load = float(net.load.p_mw.sum()) if len(net.load) else 0.0
    rng = np.random.default_rng(NETWORK_SEED)
    coeff = 1.0 + 0.5 * rng.random(len(elements))
    coeff += 0.4 * np.arange(len(elements)) / max(1, len(elements) - 1)
    inverse = 1.0 / coeff
    unconstrained = total_load * inverse / inverse.sum()
    slack = max(50.0, 0.25 * total_load)
    for pos, (element_type, element) in enumerate(elements):
        pmin = max(0.0, float(unconstrained[pos] - slack))
        pmax = float(unconstrained[pos] + slack)
        table = net.gen if element_type == "gen" else net.ext_grid
        table.at[element, "min_p_mw"] = pmin
        table.at[element, "max_p_mw"] = pmax
        pp.create_poly_cost(
            net,
            element=element,
            et=element_type,
            cp2_eur_per_mw2=float(coeff[pos]),
            cp1_eur_per_mw=0.0,
            cp0_eur=0.0,
        )

    pp.rundcopp(net, verbose=False)
    if not hasattr(net, "_ppc") or not isinstance(net._ppc, dict):
        raise RuntimeError("pandapower did not retain the solved PYPOWER case")
    return net


def _load_internal_buses(net, ppc_i: Dict[str, Any]) -> np.ndarray:
    """Map each pandapower load row to the internal PYPOWER bus row."""
    if not len(net.load):
        return np.zeros(0, dtype=np.int64)
    lookup = np.asarray(net._pd2ppc_lookups["bus"], dtype=np.int64)
    order = ppc_i.get("order", {}).get("bus", {})
    e2i = np.asarray(order.get("e2i"), dtype=np.int64)
    if e2i.size == 0:
        raise RuntimeError("ext2int bus mapping is unavailable")
    result = []
    for pp_bus in net.load.bus.astype(int).tolist():
        external_bus = int(lookup[pp_bus])
        internal_bus = int(e2i[external_bus])
        if internal_bus < 0:
            raise RuntimeError(f"load bus {pp_bus} was removed by ext2int")
        result.append(internal_bus)
    return np.asarray(result, dtype=np.int64)


def build_network_contract() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Return pandapower net, external PPC, and audited static arrays."""
    imp = _pypower_imports()
    net = build_base_net()
    ppc = copy.deepcopy(net._ppc)
    ppc_i = imp["ext2int"](copy.deepcopy(ppc))
    bus = np.asarray(ppc_i["bus"], dtype=np.float64)
    branch = np.asarray(ppc_i["branch"], dtype=np.float64)
    gen_all = np.asarray(ppc_i["gen"], dtype=np.float64)
    gencost_all = np.asarray(ppc_i["gencost"], dtype=np.float64)
    on = np.flatnonzero(gen_all[:, imp["GEN_STATUS"]] > 0.0)
    gen = gen_all[on]
    gencost = gencost_all[on]
    ref_rows = np.flatnonzero(bus[:, imp["BUS_TYPE"]] == imp["REF"])
    ref = int(ref_rows[0]) if len(ref_rows) else 0
    base_mva = float(ppc_i["baseMVA"])
    ptdf = np.asarray(
        imp["makePTDF"](base_mva, bus, branch, ref), dtype=np.float64
    )
    offset = _flow_offset(imp["makeBdc"], base_mva, bus, branch, ptdf)

    fmax_raw = branch[:, imp["RATE_A"]].astype(np.float64)
    fmax = np.where(fmax_raw <= 0.0, 1e9, fmax_raw)
    model = gencost[:, 0].astype(int)
    ncost = gencost[:, 3].astype(int)
    if not np.all(model == imp["POLYNOMIAL"]) or not np.all(ncost == 3):
        raise RuntimeError("the smoke contract requires quadratic polynomial costs")
    coeffs = gencost[:, 4:7].astype(np.float64)
    c2, c1, c0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    if np.any(c2 <= 0.0):
        raise RuntimeError("generation objective is not strictly convex")

    line_lookup = net._pd2ppc_lookups.get("branch", {}).get("line")
    if not line_lookup or len(line_lookup) != 2:
        raise RuntimeError("pandapower line-to-branch lookup is unavailable")
    line_start, line_end = map(int, line_lookup)
    line_pp_indices = np.asarray(net.line.index.tolist(), dtype=np.int64)
    if line_end - line_start != len(line_pp_indices):
        raise RuntimeError("pandapower line lookup size mismatch")
    line_branch_rows = np.arange(line_start, line_end, dtype=np.int64)

    static = {
        "base_mva": base_mva,
        "nbus": int(bus.shape[0]),
        "ngen": int(gen.shape[0]),
        "nbranch": int(branch.shape[0]),
        "bus": bus,
        "branch": branch,
        "gen": gen,
        "gencost": gencost,
        "bus_of_gen": gen[:, imp["GEN_BUS"]].astype(np.int64),
        "pmin": gen[:, imp["PMIN"]].astype(np.float64),
        "pmax": gen[:, imp["PMAX"]].astype(np.float64),
        "base_pg": gen[:, imp["PG"]].astype(np.float64),
        "base_pd": bus[:, imp["PD"]].astype(np.float64),
        "gs_bus": bus[:, imp["GS"]].astype(np.float64),
        "ptdf": ptdf,
        "flow_offset": offset,
        "fmax": fmax,
        "fmax_raw": fmax_raw,
        "c2": c2,
        "c1": c1,
        "c0": c0,
        "branch_from": branch[:, imp["F_BUS"]].astype(np.int64),
        "branch_to": branch[:, imp["T_BUS"]].astype(np.int64),
        "line_pp_indices": line_pp_indices,
        "line_branch_rows": line_branch_rows,
        "load_internal_buses": _load_internal_buses(net, ppc_i),
        "load_base_p_mw": net.load.p_mw.astype(float).to_numpy(copy=True),
        "phase_shift_nonzero_count": int(
            np.count_nonzero(np.abs(branch[:, imp["SHIFT"]]) > 1e-12)
        ),
        "ref_bus": ref,
    }
    return net, ppc, static


def discover_line_security_set(
    static: Dict[str, Any], denominator_tol: float = 1e-8
) -> Dict[str, Any]:
    """Build all finite non-islanding LODF columns for pandapower lines."""
    h = static["ptdf"][:, static["branch_from"]]
    h = h - static["ptdf"][:, static["branch_to"]]
    valid_rows: List[int] = []
    valid_pp: List[int] = []
    columns: List[np.ndarray] = []
    excluded: List[Dict[str, Any]] = []
    for pp_idx, row in zip(static["line_pp_indices"], static["line_branch_rows"]):
        denom = float(1.0 - h[int(row), int(row)])
        if not np.isfinite(denom) or abs(denom) <= denominator_tol:
            excluded.append(
                {
                    "line_pp_idx": int(pp_idx),
                    "branch_row": int(row),
                    "reason": "islanding_or_singular_lodf_denominator",
                    "denominator": denom,
                }
            )
            continue
        col = h[:, int(row)] / denom
        col = np.asarray(col, dtype=np.float64)
        col[int(row)] = -1.0
        if not np.all(np.isfinite(col)):
            excluded.append(
                {
                    "line_pp_idx": int(pp_idx),
                    "branch_row": int(row),
                    "reason": "nonfinite_lodf_column",
                    "denominator": denom,
                }
            )
            continue
        valid_pp.append(int(pp_idx))
        valid_rows.append(int(row))
        columns.append(col)
    if not columns:
        raise RuntimeError("no valid line contingencies were discovered")
    lodf = np.stack(columns, axis=1)
    payload = [
        {"line_pp_idx": pp, "branch_row": row}
        for pp, row in zip(valid_pp, valid_rows)
    ]
    return {
        "line_pp_indices": np.asarray(valid_pp, dtype=np.int64),
        "branch_rows": np.asarray(valid_rows, dtype=np.int64),
        "lodf_columns": lodf,
        "excluded": excluded,
        "digest": stable_json_digest(payload),
    }


def select_smoke_security_set(
    static: Dict[str, Any], full_security: Dict[str, Any], count: int = 32
) -> Dict[str, Any]:
    """Select a deterministic fixed smoke set by base-flow loading.

    This does not claim full IEEE 300 N-1 coverage.  It creates a declared
    K-contingency preventive SCOPF whose K outages are all enforced together.
    The full valid set remains available for an explicit infeasibility gate.
    """
    if count < 1 or count > len(full_security["branch_rows"]):
        raise ValueError("smoke contingency count is outside the eligible set")
    effective = static["base_pd"] + static["gs_bus"]
    base_flow = (
        static["ptdf"][:, static["bus_of_gen"]] @ static["base_pg"]
        + static["flow_offset"]
        - static["ptdf"] @ effective
    )
    scores = np.abs(base_flow[full_security["branch_rows"]])
    order = np.argsort(-scores, kind="stable")[:count]
    rows = full_security["branch_rows"][order].copy()
    pp_indices = full_security["line_pp_indices"][order].copy()
    lodf = full_security["lodf_columns"][:, order].copy()
    payload = [
        {"line_pp_idx": int(pp), "branch_row": int(row)}
        for pp, row in zip(pp_indices, rows)
    ]
    selected_positions = set(int(x) for x in order.tolist())
    screened_out = [
        {
            "line_pp_idx": int(full_security["line_pp_indices"][pos]),
            "branch_row": int(full_security["branch_rows"][pos]),
            "reason": "eligible_but_not_in_fixed_smoke_set",
        }
        for pos in range(len(full_security["branch_rows"]))
        if pos not in selected_positions
    ]
    return {
        "line_pp_indices": pp_indices,
        "branch_rows": rows,
        "lodf_columns": lodf,
        "excluded": list(full_security["excluded"]),
        "screened_out_eligible": screened_out,
        "eligible_count": int(len(full_security["branch_rows"])),
        "selection_policy": (
            f"top_{count}_eligible_pandapower_lines_by_absolute_base_opf_flow"
        ),
        "digest": stable_json_digest(payload),
    }


def sample_pd_bus(
    static: Dict[str, Any], rng: np.random.Generator, low: float, high: float
) -> Tuple[np.ndarray, np.ndarray]:
    factors = rng.uniform(low, high, size=len(static["load_base_p_mw"]))
    pd = np.array(static["base_pd"], dtype=np.float64, copy=True)
    delta = static["load_base_p_mw"] * (factors - 1.0)
    np.add.at(pd, static["load_internal_buses"], delta)
    return pd, factors


def economic_dispatch(static: Dict[str, Any], total_demand: float) -> np.ndarray:
    pmin = static["pmin"]
    pmax = static["pmax"]
    if total_demand < pmin.sum() - 1e-8 or total_demand > pmax.sum() + 1e-8:
        raise RuntimeError("demand lies outside aggregate generator capacity")
    lo = float(np.min(2.0 * static["c2"] * pmin + static["c1"]) - 1.0)
    hi = float(np.max(2.0 * static["c2"] * pmax + static["c1"]) + 1.0)
    for _ in range(100):
        marginal = 0.5 * (lo + hi)
        p = np.clip(
            (marginal - static["c1"]) / (2.0 * static["c2"]), pmin, pmax
        )
        if p.sum() < total_demand:
            lo = marginal
        else:
            hi = marginal
    p = np.clip((hi - static["c1"]) / (2.0 * static["c2"]), pmin, pmax)
    residual = total_demand - float(p.sum())
    if abs(residual) > 1e-8:
        room = (pmax - p) if residual > 0 else (p - pmin)
        available = np.flatnonzero(room > 1e-10)
        for idx in available:
            step = np.sign(residual) * min(abs(residual), float(room[idx]))
            p[idx] += step
            residual -= step
            if abs(residual) <= 1e-9:
                break
    return p


@dataclass
class SolveResult:
    pg: np.ndarray
    objective: float
    base_flow: np.ndarray
    worst_violation_mw: float
    worst_contingency_position: int
    worst_branch_row: int
    cut_rounds: int
    active_security_rows: int
    qp_iterations: int
    solve_time_sec: float
    solver_success: bool
    solver_status: int
    solver_message: str
    balance_abs_mw: float
    generator_bound_violation_mw: float
    base_flow_violation_mw: float


def all_contingency_flows(
    base_flow: np.ndarray, security: Dict[str, Any]
) -> np.ndarray:
    rows = security["branch_rows"]
    return base_flow[None, :] + security["lodf_columns"].T * base_flow[rows, None]


def security_violation_matrix(
    contingency_flows: np.ndarray,
    fmax: np.ndarray,
    outaged_rows: np.ndarray,
) -> np.ndarray:
    violation = np.abs(contingency_flows) - fmax[None, :]
    violation[np.arange(len(outaged_rows)), outaged_rows] = -np.inf
    return violation


def objective_value(static: Dict[str, Any], pg: np.ndarray) -> float:
    return float(np.sum(static["c2"] * pg * pg + static["c1"] * pg + static["c0"]))


def solve_preventive_scopf(
    static: Dict[str, Any],
    security: Dict[str, Any],
    pd_bus: np.ndarray,
    *,
    warm_start: Optional[np.ndarray] = None,
    security_tol_mw: float = 1e-5,
    diagnostic_tie_tol_mw: float = DIAGNOSTIC_TIE_TOL_MW,
    max_cut_rounds: int = 30,
    cuts_per_round: int = 128,
) -> SolveResult:
    """Solve the convex preventive QP by exact contingency constraint generation."""
    imp = _pypower_imports()
    started = time.perf_counter()
    effective_demand = np.asarray(pd_bus, dtype=np.float64) + static["gs_bus"]
    demand_total = float(effective_demand.sum())
    gen_flow = static["ptdf"][:, static["bus_of_gen"]]
    flow_constant = static["flow_offset"] - static["ptdf"] @ effective_demand
    finite_base = np.flatnonzero(static["fmax"] < 1e8)

    balance_row = csr_matrix(np.ones((1, static["ngen"]), dtype=np.float64))
    base_rows = csr_matrix(gen_flow[finite_base])
    base_lower = -static["fmax"][finite_base] - flow_constant[finite_base]
    base_upper = static["fmax"][finite_base] - flow_constant[finite_base]
    active_pairs: List[Tuple[int, int]] = []
    active_set = set()
    if warm_start is not None:
        x0 = np.asarray(warm_start, dtype=np.float64).copy()
    elif abs(float(static["base_pg"].sum()) - demand_total) <= 1e-6:
        # The pandapower base solution is an excellent feasible start for the
        # deterministic base-demand test.  The unconstrained ED point can be
        # hundreds of MW outside the tightened case300 line limits.
        x0 = np.asarray(static["base_pg"], dtype=np.float64).copy()
    else:
        x0 = economic_dispatch(static, demand_total)
    if x0.shape != (static["ngen"],):
        raise ValueError("warm start has the wrong generator dimension")
    qp_iterations = 0

    for cut_round in range(max_cut_rounds + 1):
        matrices = [balance_row, base_rows]
        lower = [np.asarray([demand_total]), base_lower]
        upper = [np.asarray([demand_total]), base_upper]
        if active_pairs:
            extra_rows = []
            extra_lower = []
            extra_upper = []
            for cont_pos, branch_row in active_pairs:
                outage_row = int(security["branch_rows"][cont_pos])
                lodf_value = float(security["lodf_columns"][branch_row, cont_pos])
                row = gen_flow[branch_row] + lodf_value * gen_flow[outage_row]
                offset = (
                    flow_constant[branch_row]
                    + lodf_value * flow_constant[outage_row]
                )
                extra_rows.append(row)
                extra_lower.append(-static["fmax"][branch_row] - offset)
                extra_upper.append(static["fmax"][branch_row] - offset)
            matrices.append(csr_matrix(np.asarray(extra_rows)))
            lower.append(np.asarray(extra_lower))
            upper.append(np.asarray(extra_upper))
        constraint_matrix = vstack(matrices, format="csr")
        constraint_lower = np.concatenate(lower)
        constraint_upper = np.concatenate(upper)
        # SciPy SLSQP is used here instead of PYPOWER PIPS because the latter
        # is numerically unstable on the tightened case300 balance/flow scale.
        # The problem remains the same strictly convex QP with linear
        # constraints; the independent auditor uses a separate reconstruction
        # and reference-solve path.
        dense_constraints = constraint_matrix.toarray()
        equality = LinearConstraint(
            dense_constraints[:1], constraint_lower[:1], constraint_upper[:1]
        )
        inequalities = LinearConstraint(
            dense_constraints[1:], constraint_lower[1:], constraint_upper[1:]
        )
        # Scale by MW demand rather than the O(1e7) total cost.  Cost scaling
        # that is too large makes a several-unit unscaled KKT residual appear
        # tiny to SLSQP's stopping test even though primal feasibility is exact.
        objective_scale = max(1.0, demand_total)

        def fun(p: np.ndarray) -> float:
            return objective_value(static, p) / objective_scale

        def jac(p: np.ndarray) -> np.ndarray:
            return (2.0 * static["c2"] * p + static["c1"]) / objective_scale

        def run_slsqp(start: np.ndarray):
            return minimize(
                fun,
                start,
                jac=jac,
                method="SLSQP",
                bounds=Bounds(static["pmin"], static["pmax"]),
                constraints=[equality, inequalities],
                options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
            )

        result = run_slsqp(x0)
        if not bool(result.success):
            # HiGHS provides a robust, independent linear-feasibility oracle.
            # If the current security cuts are feasible, restart the QP from
            # its feasible point; if not, the preventive model is genuinely
            # infeasible for this demand under the declared limits.
            flow_a = dense_constraints[1:]
            flow_l = constraint_lower[1:]
            flow_u = constraint_upper[1:]
            lp = linprog(
                np.zeros(static["ngen"], dtype=np.float64),
                A_ub=np.vstack([flow_a, -flow_a]),
                b_ub=np.concatenate([flow_u, -flow_l]),
                A_eq=dense_constraints[:1],
                b_eq=np.asarray([demand_total]),
                bounds=list(zip(static["pmin"], static["pmax"])),
                method="highs",
            )
            if bool(lp.success):
                x0 = np.asarray(lp.x, dtype=np.float64)
                result = run_slsqp(x0)
            else:
                raise RuntimeError(
                    f"preventive SC-DCOPF is linearly infeasible at cut_round={cut_round} "
                    f"active_security_rows={len(active_pairs)}: {lp.message}"
                )
        pg = np.asarray(result.x, dtype=np.float64)
        qp_iterations += int(getattr(result, "nit", 0))
        if not bool(result.success) or not np.all(np.isfinite(pg)):
            raise RuntimeError(
                f"preventive SC-DCOPF QP failed at cut_round={cut_round} "
                f"active_security_rows={len(active_pairs)}: status={result.status} "
                f"message={result.message}"
            )

        base_flow = gen_flow @ pg + flow_constant
        contingency_flow = all_contingency_flows(base_flow, security)
        violation = security_violation_matrix(
            contingency_flow, static["fmax"], security["branch_rows"]
        )
        flat_index = int(np.argmax(violation))
        worst_pos, worst_branch = np.unravel_index(flat_index, violation.shape)
        contingency_raw_worst = float(violation[worst_pos, worst_branch])
        base_raw = np.abs(base_flow) - static["fmax"]
        base_raw_worst = float(np.max(base_raw))
        total_worst = max(0.0, contingency_raw_worst, base_raw_worst)
        diagnostic_threshold = max(base_raw_worst, contingency_raw_worst) - float(
            diagnostic_tie_tol_mw
        )
        tied_base = np.flatnonzero(base_raw >= diagnostic_threshold)
        if len(tied_base):
            # Base state is encoded as -1 and therefore comes first in the
            # documented lexicographic tie-break; branch row breaks the tie.
            reported_contingency_position = -1
            reported_branch_row = int(tied_base[0])
        else:
            tied_contingency = np.argwhere(violation >= diagnostic_threshold)
            if not len(tied_contingency):
                raise RuntimeError("failed to identify a canonical worst constraint")
            reported_contingency_position = int(tied_contingency[0, 0])
            reported_branch_row = int(tied_contingency[0, 1])
        if total_worst <= security_tol_mw:
            balance_abs = abs(float(pg.sum()) - demand_total)
            bound_violation = max(
                0.0,
                float(np.max(static["pmin"] - pg)),
                float(np.max(pg - static["pmax"])),
            )
            return SolveResult(
                pg=pg,
                objective=objective_value(static, pg),
                base_flow=base_flow,
                worst_violation_mw=total_worst,
                worst_contingency_position=reported_contingency_position,
                worst_branch_row=reported_branch_row,
                cut_rounds=cut_round,
                active_security_rows=len(active_pairs),
                qp_iterations=qp_iterations,
                solve_time_sec=time.perf_counter() - started,
                solver_success=bool(result.success),
                solver_status=int(result.status),
                solver_message=str(result.message),
                balance_abs_mw=balance_abs,
                generator_bound_violation_mw=bound_violation,
                base_flow_violation_mw=max(0.0, base_raw_worst),
            )

        order = np.argsort(violation, axis=None)[::-1]
        added = 0
        for flat in order:
            cont_pos, branch_row = np.unravel_index(int(flat), violation.shape)
            value = float(violation[cont_pos, branch_row])
            if value <= security_tol_mw:
                break
            pair = (int(cont_pos), int(branch_row))
            if pair in active_set:
                continue
            active_pairs.append(pair)
            active_set.add(pair)
            added += 1
            if added >= cuts_per_round:
                break
        if added == 0:
            raise RuntimeError(
                f"QP remained {total_worst:.6g} MW insecure after all violated cuts were active"
            )
        x0 = pg
    raise RuntimeError(f"constraint generation exceeded {max_cut_rounds} rounds")


def split_ids(n: int) -> np.ndarray:
    n_train = int(round(0.70 * n))
    n_val = int(round(0.10 * n))
    result = np.full(n, 2, dtype=np.int8)
    result[:n_train] = 0
    result[n_train : n_train + n_val] = 1
    return result


def static_hashes(static: Dict[str, Any], security: Dict[str, Any]) -> Dict[str, str]:
    return {
        "ptdf": sha256_array(static["ptdf"]),
        "flow_offset": sha256_array(static["flow_offset"]),
        "fmax": sha256_array(static["fmax"]),
        "pmin": sha256_array(static["pmin"]),
        "pmax": sha256_array(static["pmax"]),
        "cost_c2": sha256_array(static["c2"]),
        "cost_c1": sha256_array(static["c1"]),
        "cost_c0": sha256_array(static["c0"]),
        "bus_of_gen": sha256_array(static["bus_of_gen"]),
        "gs_bus": sha256_array(static["gs_bus"]),
        "security_lodf_columns": sha256_array(security["lodf_columns"]),
    }
