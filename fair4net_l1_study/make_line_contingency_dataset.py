#!/usr/bin/env python3
"""Generate a GS-aware, disjoint N-1 line-contingency dataset.

The saved ``pd_bus`` is conventional demand only.  Fixed active shunts are
stored once as ``gs_bus`` and must be added by the Fair4Net operator.  Each
sample stores ``cont_line_idx`` in the base PYPOWER branch ordering (``-1``
means no outage) and a precomputed ``split`` (0=train, 1=validation,
2=test).  Outage identities are disjoint across the three splits.

This generator intentionally stores compact scenario labels rather than a
per-sample PTDF tensor.  The trainer applies the selected outage through the
base PTDF plus its LODF column, which keeps large-network files manageable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from pandapower.auxiliary import OPFNotConverged
except ImportError:  # pragma: no cover - compatibility with older pandapower
    from pandapower.optimal_powerflow import OPFNotConverged

import dc_opf_pandapower_case300_9k as shared
import deq_trainer_fair4net as tr


def _pypower_imports():
    try:
        from pypower.ext2int import ext2int
        from pypower.idx_bus import PD
        from pypower.idx_gen import PG
        from pypower.makeLODF import makeLODF
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.idx_bus import PD
        from pandapower.pypower.idx_gen import PG
        from pandapower.pypower.makeLODF import makeLODF
    return ext2int, PD, PG, makeLODF


def _branch_keys(ppc: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[float, ...]]]:
    """Return the internal branch matrix and stable matching keys."""
    ext2int, *_ = _pypower_imports()
    ppc_i = ext2int(copy.deepcopy(ppc))
    branch = np.asarray(ppc_i["branch"], dtype=float)
    # pandapower may retain an outaged branch in the PYPOWER matrix with
    # BR_STATUS=0 rather than deleting the row.  Branch matching and PTDF
    # construction must use the active topology, otherwise the outage looks
    # like an unmatched parallel branch and every candidate is rejected.
    if branch.shape[1] > 10:
        branch = branch[branch[:, 10] > 0.0]
    # Include only topology/nameplate/status columns.  Do not include solved
    # flow columns (PF/QF/PT/QT, indices 13-16): those values legitimately
    # change after an outage and would make every surviving branch look new.
    # RATE_A/B/C and ANG limits make the key robust to re-embedding while the
    # endpoint/impedance fields distinguish parallel branches.
    cols = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    keys: List[Tuple[float, ...]] = []
    for row in branch:
        vals = []
        for c in cols:
            x = float(row[c])
            vals.append(round(0.0 if abs(x) < 1e-12 else x, 10))
        keys.append(tuple(vals))
    return branch, keys


def _map_current_to_base(base_keys: List[Tuple[float, ...]], current_keys: List[Tuple[float, ...]]) -> np.ndarray:
    locs: Dict[Tuple[float, ...], List[int]] = {}
    for i, key in enumerate(base_keys):
        locs.setdefault(key, []).append(i)
    used = {key: 0 for key in locs}
    out = np.full(len(current_keys), -1, dtype=np.int64)
    for j, key in enumerate(current_keys):
        if key not in locs:
            raise ValueError("current branch cannot be matched to base branch")
        ptr = used[key]
        if ptr >= len(locs[key]):
            raise ValueError("parallel branch-key multiplicity exceeded")
        out[j] = locs[key][ptr]
        used[key] = ptr + 1
    return out


def _build_base_net():
    net = shared.load_ieee_case(shared.CASE_NAME)
    shared.scale_loads(net, shared.LOAD_SCALE)
    shared.ensure_controllable(net)
    shared.set_voltage_limits(net, *shared.VM_BOUNDS)
    shared.scale_line_and_trafo_limits(net, shared.LINE_LIMIT_SCALE)
    shared.add_heterogeneous_quadratic_costs_and_bounds(net, shared.GEN_MARGIN_MW, shared.SEED)
    return net


def _solve_and_extract(net) -> Dict[str, np.ndarray]:
    ext2int, PD, PG, _ = _pypower_imports()
    ppc = tr.build_ppc_from_net(net)
    ppc_i = ext2int(copy.deepcopy(ppc))
    bus = np.asarray(ppc_i["bus"], dtype=float)
    gen = np.asarray(ppc_i["gen"], dtype=float)
    branch = np.asarray(ppc_i["branch"], dtype=float)
    active = np.ones(branch.shape[0], dtype=bool)
    if branch.shape[1] > 10:
        active = branch[:, 10] > 0.0
    return {
        "pd_bus": bus[:, PD].astype(np.float32),
        "pg": gen[:, PG].astype(np.float32),
        # Keep flow labels in the same active-branch ordering used by
        # _branch_keys; the outaged branch is represented by the zero-filled
        # base-sized arrays assembled by the caller.
        "pf": branch[active, 13].astype(np.float32),
        "pt": branch[active, 15].astype(np.float32),
        "ppc": ppc,
    }


def _active_line_indices(net) -> List[int]:
    if len(net.line) == 0:
        return []
    if "in_service" in net.line.columns:
        return [int(i) for i in net.line.index[net.line.in_service.astype(bool)].tolist()]
    return [int(i) for i in net.line.index.tolist()]


def _line_loading_order(net) -> List[int]:
    lines = _active_line_indices(net)
    if not lines:
        return []
    if hasattr(net, "res_line") and len(net.res_line):
        vals = []
        for idx in lines:
            row = net.res_line.loc[idx]
            p_from = float(row.get("p_from_mw", 0.0))
            p_to = float(row.get("p_to_mw", 0.0))
            vals.append((max(abs(p_from), abs(p_to)), idx))
        return [idx for _, idx in sorted(vals, reverse=True)]
    return lines


def _valid_lodf_columns(ppc: Dict[str, Any], static: Dict[str, Any]) -> np.ndarray:
    """Return finite LODF columns for the base internal branch ordering."""
    ext2int, *_ = _pypower_imports()
    _, _, _, makeLODF = _pypower_imports()
    ppc_i = ext2int(copy.deepcopy(ppc))
    branch = np.asarray(ppc_i["branch"], dtype=float)
    try:
        lodf = np.asarray(makeLODF(branch, np.asarray(static["ptdf_dense"], dtype=float)), dtype=float)
    except Exception:
        return np.zeros(branch.shape[0], dtype=bool)
    return np.isfinite(lodf).all(axis=0)


def _direct_lodf_column(
    base_ppc: Dict[str, Any],
    current_ppc: Dict[str, Any],
    static: Dict[str, Any],
    base_row: int,
) -> np.ndarray:
    """Derive one LODF column from the post-outage topology PTDF.

    Some pandapower/PYPOWER versions return a numerically unusable matrix from
    ``makeLODF`` for these cases (all columns contain division-by-zero values).
    The outage identity is still well-defined for a non-islanding line: compute
    the post-outage PTDF directly, evaluate both PTDFs on the outaged branch's
    endpoint injection, and divide by the corresponding base flow.  This also
    gives us an explicit finite-column gate before a line enters the dataset.
    """
    try:
        from pypower.ext2int import ext2int
        from pypower.idx_brch import F_BUS, T_BUS
        from pypower.makePTDF import makePTDF
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.idx_brch import F_BUS, T_BUS
        from pandapower.pypower.makePTDF import makePTDF

    base_i = ext2int(copy.deepcopy(base_ppc))
    cur_i = ext2int(copy.deepcopy(current_ppc))
    base_bus = np.asarray(base_i["bus"], dtype=float)
    cur_bus = np.asarray(cur_i["bus"], dtype=float)
    base_branch = np.asarray(base_i["branch"], dtype=float)
    cur_branch = np.asarray(cur_i["branch"], dtype=float)
    cur_active = np.ones(cur_branch.shape[0], dtype=bool)
    if cur_branch.shape[1] > 10:
        cur_active = cur_branch[:, 10] > 0.0
    cur_branch = cur_branch[cur_active]
    if cur_bus.shape != base_bus.shape:
        raise ValueError("outage changed the internal bus set")

    if cur_branch.shape[0] != base_branch.shape[0] - 1:
        raise ValueError("outage did not remove exactly one base branch")
    cur_to_base = np.delete(np.arange(base_branch.shape[0], dtype=np.int64), base_row)
    # pandapower preserves PYPOWER branch order when removing one line.  Check
    # the static columns explicitly so a converter-version change fails fast.
    if not np.allclose(
        cur_branch[:, :13], base_branch[cur_to_base, :13], atol=1e-10, rtol=0.0
    ):
        raise ValueError("post-outage branch ordering differs from base order")

    f = int(base_branch[base_row, F_BUS])
    t = int(base_branch[base_row, T_BUS])
    probe = np.zeros(base_bus.shape[0], dtype=float)
    probe[f] = 1.0
    probe[t] = -1.0
    base_ptdf = np.asarray(static["ptdf_dense"], dtype=float)
    denom = float(base_ptdf[base_row] @ probe)
    if not np.isfinite(denom) or abs(denom) < 1e-8 or abs(denom - 1.0) < 1e-8:
        raise ValueError("outage is islanding or has an ill-conditioned transfer denominator")

    base_mva = float(base_i["baseMVA"])
    base_probe = base_ptdf @ probe
    cur_ptdf = np.asarray(makePTDF(base_mva, cur_bus, cur_branch, 0), dtype=float)
    if cur_ptdf.shape != (len(cur_to_base), base_bus.shape[0]):
        raise ValueError(f"unexpected post-outage PTDF shape {cur_ptdf.shape}")
    cur_probe = np.zeros_like(base_probe)
    cur_probe[cur_to_base] = cur_ptdf @ probe
    col = (cur_probe - base_probe) / denom
    # Standard LODF convention.  This makes the outaged PTDF row and affine
    # flow offset exactly zero under H_k = H + L[:,k] H[k,:].
    col[base_row] = -1.0
    if not np.all(np.isfinite(col)):
        raise ValueError("post-outage LODF column is non-finite")
    return col.astype(np.float32)


def discover_line_candidates(
    base_net,
    base_ppc: Dict[str, Any],
    static: Dict[str, Any],
    count: int,
    seed: int,
) -> Tuple[List[Dict[str, int]], np.ndarray]:
    """Map pandapower line IDs to valid base internal branch rows.

    Candidate discovery solves each selected line outage once.  This is a
    one-time setup cost and prevents islanding/nonfinite-L ODF scenarios from
    entering the dataset.
    """
    base_branch, _ = _branch_keys(base_ppc)
    branch_lookup = getattr(base_net, "_pd2ppc_lookups", {}).get("branch", {}).get("line")
    if not branch_lookup or len(branch_lookup) != 2:
        raise RuntimeError("pandapower line-to-PYPOWER branch lookup is unavailable")
    line_start, line_end = map(int, branch_lookup)
    line_indices = [int(x) for x in base_net.line.index.tolist()]
    if line_end - line_start != len(line_indices):
        raise RuntimeError("pandapower line lookup size does not match net.line")
    line_to_base_row = {pp_idx: line_start + pos for pos, pp_idx in enumerate(line_indices)}
    ordered = _line_loading_order(base_net)
    # Add a deterministic random tail so top-loading candidates do not exhaust
    # the pool when a stressed line islands the network.
    rng = np.random.default_rng(seed)
    tail = ordered[:]
    rng.shuffle(tail)
    ordered = list(dict.fromkeys(ordered + tail))

    found: List[Dict[str, int]] = []
    lodf_columns = np.zeros((int(static["nbranch"]), int(static["nbranch"])), dtype=np.float32)
    failure_reasons: Counter[str] = Counter()
    discovery_start = time.perf_counter()
    for attempt, pp_idx in enumerate(ordered, start=1):
        if len(found) >= count:
            break
        trial = copy.deepcopy(base_net)
        trial.line.at[int(pp_idx), "in_service"] = False
        try:
            base_row = int(line_to_base_row[int(pp_idx)])
            cur = _solve_and_extract(trial)
            lodf_col = _direct_lodf_column(base_ppc, cur["ppc"], static, base_row)
            found.append({"line_pp_idx": int(pp_idx), "cont_line_idx": base_row})
            lodf_columns[:, base_row] = lodf_col
            print(
                f"[candidate-progress] attempt={attempt} accepted={len(found)}/{count} "
                f"line_pp_idx={int(pp_idx)} cont_line_idx={base_row} status=accepted "
                f"elapsed_sec={time.perf_counter() - discovery_start:.1f}",
                flush=True,
            )
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            failure_reasons[reason] += 1
            print(
                f"[candidate-progress] attempt={attempt} accepted={len(found)}/{count} "
                f"line_pp_idx={int(pp_idx)} status=rejected reason={reason} "
                f"elapsed_sec={time.perf_counter() - discovery_start:.1f}",
                flush=True,
            )
            continue
    if len(found) < max(3, count // 2):
        detail = "; ".join(f"{k} ({v})" for k, v in failure_reasons.most_common(5))
        raise RuntimeError(
            f"only discovered {len(found)} valid line candidates; requested {count}; "
            f"top rejection reasons: {detail or 'none'}"
        )
    return found, lodf_columns


def _split_pools(candidates: List[Dict[str, int]], seed: int) -> Dict[str, List[Dict[str, int]]]:
    rng = np.random.default_rng(seed)
    items = list(candidates)
    rng.shuffle(items)
    n = len(items)
    n_train = max(1, int(round(0.70 * n)))
    n_val = max(1, int(round(0.10 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}


def _counts(n: int) -> Tuple[int, int, int]:
    n_train = int(round(0.70 * n))
    n_val = int(round(0.10 * n))
    return n_train, n_val, n - n_train - n_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta", default="", type=Path)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-line-candidates", type=int, default=8)
    ap.add_argument("--p-none", type=float, default=0.25)
    ap.add_argument("--scale-low", type=float, default=0.9)
    ap.add_argument("--scale-high", type=float, default=1.1)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="print one flushed progress record every this many globally accepted rows",
    )
    ap.add_argument(
        "--ensure-candidate-coverage",
        action="store_true",
        help="force every declared split-specific outage candidate into at least one accepted row",
    )
    ap.add_argument(
        "--max-attempt-factor",
        type=float,
        default=2.0,
        help=(
            "maximum attempted OPF draws divided by requested accepted rows, "
            "applied separately to each split"
        ),
    )
    ap.add_argument("--no-flow-labels", action="store_true", help="omit diagnostic pf/pt arrays for large datasets")
    args = ap.parse_args()
    if args.N < 3:
        raise ValueError("N must be at least 3")
    if not (0.0 <= args.p_none < 1.0):
        raise ValueError("p-none must be in [0,1)")
    if args.max_attempt_factor < 1.0:
        raise ValueError("max-attempt-factor must be at least 1")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")

    rng = np.random.default_rng(args.seed)
    base_net = _build_base_net()
    # Build on the actual object so pandapower's authoritative element-to-PPC
    # lookup is retained for distinct mapping of parallel line elements.
    base_ppc = tr.build_ppc_from_net(base_net)
    static = tr.extract_static_mats(base_ppc)
    base_branch, _ = _branch_keys(base_ppc)
    candidates, lodf_columns = discover_line_candidates(
        base_net, base_ppc, static, args.n_line_candidates, args.seed
    )
    pools = _split_pools(candidates, args.seed)
    n_train, n_val, n_test = _counts(args.N)
    split_sizes = [n_train, n_val, n_test]
    split_names = ["train", "val", "test"]

    pd_rows: List[np.ndarray] = []
    pg_rows: List[np.ndarray] = []
    cont_rows: List[int] = []
    pp_rows: List[int] = []
    split_rows: List[int] = []
    pf_rows: List[np.ndarray] = []
    pt_rows: List[np.ndarray] = []
    balance_rows: List[float] = []
    attempts_by_split: Dict[str, int] = {}
    rejections_by_split: Dict[str, int] = {}
    rejections_by_cont_line_idx: Counter[str] = Counter()

    for split_id, (name, count) in enumerate(zip(split_names, split_sizes)):
        pool = pools[name]
        if not pool:
            raise RuntimeError(f"empty outage pool for split={name}")
        accepted = 0
        attempts = 0
        max_attempts = max(count, int(np.ceil(args.max_attempt_factor * count)))
        while accepted < count:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"split={name} accepted only {accepted}/{count} rows after "
                    f"{attempts} attempts; OPF rejections={attempts - accepted}"
                )
            attempts += 1
            factors = rng.uniform(args.scale_low, args.scale_high, size=len(base_net.load)).astype(float)
            net = copy.deepcopy(base_net)
            if len(net.load):
                net.load.loc[:, "p_mw"] = net.load.p_mw.values * factors
                if "q_mvar" in net.load.columns:
                    net.load.loc[:, "q_mvar"] = net.load.q_mvar.values * factors
            if args.ensure_candidate_coverage and accepted < len(pool):
                # The smoke audit must exercise every declared outage column.
                # If OPF rejects this draw, accepted does not advance and the
                # same candidate is retried with a fresh deterministic load.
                chosen = pool[accepted]
                cont_idx = int(chosen["cont_line_idx"])
                pp_idx = int(chosen["line_pp_idx"])
                net.line.at[pp_idx, "in_service"] = False
            elif rng.random() < args.p_none:
                cont_idx = -1
                pp_idx = -1
            else:
                chosen = pool[int(rng.integers(0, len(pool)))]
                cont_idx = int(chosen["cont_line_idx"])
                pp_idx = int(chosen["line_pp_idx"])
                net.line.at[pp_idx, "in_service"] = False
            try:
                sol = _solve_and_extract(net)
            except OPFNotConverged:
                # Supervised labels exist only for feasible/converged OPF
                # scenarios. Reject the complete load/outage draw and sample a
                # fresh one from the same split-specific outage pool. The
                # seeded RNG and recorded counters make this deterministic and
                # auditable. All unrelated exceptions remain fatal.
                rejections_by_cont_line_idx[str(cont_idx)] += 1
                rejected = attempts - accepted
                if rejected <= 3 or rejected % 25 == 0:
                    print(
                        f"[sample-reject] split={name} attempt={attempts} "
                        f"accepted={accepted}/{count} cont_line_idx={cont_idx} "
                        "reason=OPFNotConverged",
                        flush=True,
                    )
                continue
            if sol["pd_bus"].shape[0] != static["nbus"] or sol["pg"].shape[0] != static["ngen"]:
                raise RuntimeError("contingency changed bus/generator dimensions")
            if cont_idx < 0:
                cur_to_base = np.arange(len(base_branch), dtype=np.int64)
            else:
                cur_to_base = np.delete(np.arange(len(base_branch), dtype=np.int64), cont_idx)
            if sol["pf"].shape[0] != cur_to_base.shape[0]:
                raise RuntimeError("post-outage flow labels do not match the base branch ordering")
            pd_rows.append(sol["pd_bus"])
            pg_rows.append(sol["pg"])
            cont_rows.append(cont_idx)
            pp_rows.append(pp_idx)
            split_rows.append(split_id)
            if not args.no_flow_labels:
                pf_base = np.zeros(len(base_branch), dtype=np.float32)
                pt_base = np.zeros(len(base_branch), dtype=np.float32)
                pf_base[cur_to_base] = sol["pf"]
                pt_base[cur_to_base] = sol["pt"]
                pf_rows.append(pf_base)
                pt_rows.append(pt_base)
            # Accumulate large-grid power balances in float64.  NumPy's
            # float32 reduction can create a false 0.01--0.02 MW imbalance
            # when thousands of stored float32 bus values are summed.
            balance_rows.append(
                float(
                    sol["pg"].astype(np.float64).sum()
                    - sol["pd_bus"].astype(np.float64).sum()
                    - np.asarray(static["gs_bus"], dtype=np.float64).sum()
                )
            )
            accepted += 1
            global_accepted = len(pd_rows)
            if global_accepted == args.N or global_accepted % args.progress_every == 0:
                print(
                    f"[sample-progress] global={global_accepted}/{args.N} "
                    f"split={name} split_accepted={accepted}/{count} "
                    f"split_attempts={attempts} split_rejected={attempts - accepted}",
                    flush=True,
                )
        attempts_by_split[name] = attempts
        rejections_by_split[name] = attempts - accepted

    pd_bus = np.stack(pd_rows).astype(np.float32)
    pg = np.stack(pg_rows).astype(np.float32)
    # The flow labels are diagnostic only; retaining them makes the smoke audit
    # able to confirm the outaged branch row is zero in the solved labels.
    pf = np.stack(pf_rows).astype(np.float32) if not args.no_flow_labels else None
    pt = np.stack(pt_rows).astype(np.float32) if not args.no_flow_labels else None
    split = np.asarray(split_rows, dtype=np.int8)
    cont_line_idx = np.asarray(cont_rows, dtype=np.int64)
    cont_line_pp_idx = np.asarray(pp_rows, dtype=np.int64)
    balance = np.asarray(balance_rows, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_arrays = dict(
        pd_bus=pd_bus,
        pg=pg,
        cont_line_idx=cont_line_idx,
        cont_line_pp_idx=cont_line_pp_idx,
        split=split,
        gs_bus=np.asarray(static["gs_bus"], dtype=np.float32),
        pd_bus_includes_gs=np.int8(0),
        dataset_schema_version=np.int16(1),
        dataset_schema_name=np.asarray("fair4net_line_contingency"),
        case_name=np.asarray(shared.CASE_NAME),
        demand_convention=np.asarray("effective_demand=pd_bus+gs_bus"),
        gs_total_mw=np.float32(np.sum(static["gs_bus"])),
        lodf=lodf_columns,
    )
    if pf is not None and pt is not None:
        save_arrays.update(pf=pf, pt=pt)
    np.savez_compressed(args.out, **save_arrays)
    meta_path = args.meta if str(args.meta) else args.out.with_suffix(".meta.json")
    meta = {
        "case_name": shared.CASE_NAME,
        "dataset_schema_name": "fair4net_line_contingency",
        "dataset_schema_version": 1,
        "N": int(args.N),
        "seed": int(args.seed),
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "line_outage_probability": float(1.0 - args.p_none),
        "progress_every": int(args.progress_every),
        "ensure_candidate_coverage": bool(args.ensure_candidate_coverage),
        "line_index_contract": "cont_line_idx is base PYPOWER internal branch row; -1 means no outage",
        "line_pp_index_contract": "cont_line_pp_idx is pandapower net.line table index; -1 means no outage",
        "flow_model_contract": tr.FLOW_MODEL_CONTRACT,
        "demand_convention": "pd_bus is conventional load; effective demand=pd_bus+gs_bus",
        "gs_total_mw": float(np.sum(static["gs_bus"])),
        "line_candidate_pools": pools,
        "base_nbranch": int(static["nbranch"]),
        "base_nbus": int(static["nbus"]),
        "base_ngen": int(static["ngen"]),
        "balance_abs_mean_MW": float(np.mean(np.abs(balance))),
        "balance_abs_max_MW": float(np.max(np.abs(balance))),
        "balance_abs_p95_MW": float(np.percentile(np.abs(balance), 95)),
        "outaged_label_rows_checked": int(np.count_nonzero(cont_line_idx >= 0)),
        "opf_sampling": {
            "policy": "reject_and_resample_OPFNotConverged_only",
            "max_attempt_factor": float(args.max_attempt_factor),
            "accepted_total": int(args.N),
            "attempts_total": int(sum(attempts_by_split.values())),
            "rejections_total": int(sum(rejections_by_split.values())),
            "rejection_rate": float(
                sum(rejections_by_split.values()) / max(1, sum(attempts_by_split.values()))
            ),
            "attempts_by_split": attempts_by_split,
            "rejections_by_split": rejections_by_split,
            "rejections_by_cont_line_idx": dict(sorted(rejections_by_cont_line_idx.items())),
        },
        "notes": "No per-sample PTDF tensor is stored; the trainer applies base PTDF+LODF for cont_line_idx.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(args.out), "meta": str(meta_path), **{k: meta[k] for k in ["case_name", "N", "split_counts", "gs_total_mw", "balance_abs_max_MW"]}}, indent=2))


if __name__ == "__main__":
    main()
