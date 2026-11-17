#!/usr/bin/env python3
"""Generate supervised DC-OPF data with clean, line, and generator scenarios.

Every row contains exactly one of three mutually exclusive scenario classes:
clean (no outage), one non-islanding line outage, or one generator outage.  Line
and generator outage identities are partitioned independently across
train/validation/test, so an outage identity used for training cannot occur in
validation or test.  Clean load draws are present in every split.

``cont_line_idx`` uses the base internal PYPOWER branch order and
``cont_gen_idx`` uses the base internal PYPOWER generator order.  ``-1`` means
that outage type is inactive.  Generator outages are imposed by fixing the
corresponding pandapower controllable element to exactly 0 MW while leaving it
in service; this preserves the base generator dimension and ordering.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from pandapower.auxiliary import OPFNotConverged
except ImportError:  # pragma: no cover
    from pandapower.optimal_powerflow import OPFNotConverged

import dc_opf_pandapower_case300_9k as shared
import deq_trainer_fair4net as tr
import make_line_contingency_dataset as line_ds


TYPE_CLEAN = 0
TYPE_LINE = 1
TYPE_GENERATOR = 2
TYPE_NAMES = {TYPE_CLEAN: "clean", TYPE_LINE: "line", TYPE_GENERATOR: "generator"}


def _generator_indices():
    try:
        from pypower.ext2int import ext2int
        from pypower.idx_gen import PMAX, PMIN
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.idx_gen import PMAX, PMIN
    return ext2int, PMIN, PMAX


def _generator_records(base_net, base_ppc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return controllable pandapower elements and their solved base dispatch."""
    del base_ppc  # the authoritative internal row is detected after each outage
    records: List[Dict[str, Any]] = []
    for element_type, result_name in (("ext_grid", "res_ext_grid"), ("gen", "res_gen")):
        table = getattr(base_net, element_type)
        result = getattr(base_net, result_name)
        for element_idx in table.index.tolist():
            records.append({
                "element_type": element_type,
                "element_index": int(element_idx),
                "base_pg_MW": float(result.at[element_idx, "p_mw"]),
            })
    return records


def _apply_generator_outage(net, record: Dict[str, Any]) -> None:
    table = getattr(net, str(record["element_type"]))
    idx = int(record["element_index"])
    table.at[idx, "min_p_mw"] = 0.0
    table.at[idx, "max_p_mw"] = 0.0


def discover_generator_candidates(
    base_net,
    base_ppc: Dict[str, Any],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Find generator outages with a feasible base-load DC-OPF label."""
    records = _generator_records(base_net, base_ppc)
    ranked = sorted(records, key=lambda row: abs(float(row["base_pg_MW"])), reverse=True)
    # Stable dispatch ranking stresses the candidate screen first; every
    # controllable element is still visited if high-output outages are infeasible.
    del seed
    ordered = ranked

    ext2int, PMIN, PMAX = _generator_indices()
    base_i = ext2int(copy.deepcopy(base_ppc))
    base_gen = np.asarray(base_i["gen"], dtype=float)
    base_zero = set(
        np.flatnonzero(
            (np.abs(base_gen[:, PMIN]) <= 1e-7)
            & (np.abs(base_gen[:, PMAX]) <= 1e-7)
        ).tolist()
    )

    found: List[Dict[str, Any]] = []
    failures: Counter[str] = Counter()
    start = time.perf_counter()
    for attempt, record in enumerate(ordered, start=1):
        if len(found) >= count:
            break
        trial = copy.deepcopy(base_net)
        try:
            _apply_generator_outage(trial, record)
            sol = line_ds._solve_and_extract(trial)
            if sol["pg"].shape[0] != len(records):
                raise ValueError("generator outage changed the internal generator dimension")
            trial_i = ext2int(copy.deepcopy(sol["ppc"]))
            trial_gen = np.asarray(trial_i["gen"], dtype=float)
            zero_rows = set(
                np.flatnonzero(
                    (np.abs(trial_gen[:, PMIN]) <= 1e-7)
                    & (np.abs(trial_gen[:, PMAX]) <= 1e-7)
                ).tolist()
            ) - base_zero
            if len(zero_rows) != 1:
                raise ValueError(
                    f"outage produced {len(zero_rows)} newly zero-bounded internal rows: "
                    f"{sorted(zero_rows)}"
                )
            idx = int(next(iter(zero_rows)))
            if abs(float(sol["pg"][idx])) > 1e-3:
                raise ValueError(
                    f"outaged generator label is {float(sol['pg'][idx]):.6g} MW rather than zero"
                )
            if any(int(row["cont_gen_idx"]) == idx for row in found):
                raise ValueError(f"duplicate internal generator row detected: {idx}")
            accepted_record = dict(record)
            accepted_record["cont_gen_idx"] = idx
            found.append(accepted_record)
            print(
                f"[gen-candidate-progress] attempt={attempt} accepted={len(found)}/{count} "
                f"cont_gen_idx={idx} element={record['element_type']}[{record['element_index']}] "
                f"status=accepted elapsed_sec={time.perf_counter() - start:.1f}",
                flush=True,
            )
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            failures[reason] += 1
            print(
                f"[gen-candidate-progress] attempt={attempt} accepted={len(found)}/{count} "
                f"element={record['element_type']}[{record['element_index']}] "
                f"status=rejected reason={reason} "
                f"elapsed_sec={time.perf_counter() - start:.1f}",
                flush=True,
            )
    if len(found) != count:
        detail = "; ".join(f"{k} ({v})" for k, v in failures.most_common(5))
        raise RuntimeError(
            f"only discovered {len(found)} valid generator candidates; requested {count}; "
            f"top rejection reasons: {detail or 'none'}"
        )
    return found


def _split_pools(items: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(0.70 * n)))
    n_val = max(1, int(round(0.10 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def _mandatory_schedule(
    line_pool: List[Dict[str, Any]], gen_pool: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any] | None]]:
    return (
        [(TYPE_CLEAN, None)]
        + [(TYPE_LINE, row) for row in line_pool]
        + [(TYPE_GENERATOR, row) for row in gen_pool]
    )


def _choose_random_scenario(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    line_pool: List[Dict[str, Any]],
    gen_pool: List[Dict[str, Any]],
) -> Tuple[int, Dict[str, Any] | None]:
    kind = int(rng.choice(np.asarray([TYPE_CLEAN, TYPE_LINE, TYPE_GENERATOR]), p=probabilities))
    if kind == TYPE_LINE:
        return kind, line_pool[int(rng.integers(0, len(line_pool)))]
    if kind == TYPE_GENERATOR:
        return kind, gen_pool[int(rng.integers(0, len(gen_pool)))]
    return kind, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta", default="", type=Path)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-line-candidates", type=int, default=8)
    ap.add_argument("--n-gen-candidates", type=int, default=8)
    ap.add_argument("--p-clean", type=float, default=0.25)
    ap.add_argument("--p-line", type=float, default=0.375)
    ap.add_argument("--p-gen", type=float, default=0.375)
    ap.add_argument("--scale-low", type=float, default=0.9)
    ap.add_argument("--scale-high", type=float, default=1.1)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--ensure-candidate-coverage", action="store_true")
    ap.add_argument("--max-attempt-factor", type=float, default=2.0)
    ap.add_argument("--no-flow-labels", action="store_true")
    args = ap.parse_args()

    if args.N < 3:
        raise ValueError("N must be at least 3")
    if args.n_line_candidates < 3 or args.n_gen_candidates < 3:
        raise ValueError("at least three candidates of each outage type are required")
    probabilities = np.asarray([args.p_clean, args.p_line, args.p_gen], dtype=float)
    if np.any(probabilities <= 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError("p-clean, p-line, and p-gen must be positive and sum to one")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if args.max_attempt_factor < 1.0:
        raise ValueError("max-attempt-factor must be at least one")

    rng = np.random.default_rng(args.seed)
    base_net = line_ds._build_base_net()
    base_ppc = tr.build_ppc_from_net(base_net)
    static = tr.extract_static_mats(base_ppc)
    base_branch, _ = line_ds._branch_keys(base_ppc)

    line_candidates, lodf_columns = line_ds.discover_line_candidates(
        base_net, base_ppc, static, args.n_line_candidates, args.seed
    )
    if len(line_candidates) != args.n_line_candidates:
        raise RuntimeError(
            f"line candidate discovery returned {len(line_candidates)} rather than "
            f"the requested {args.n_line_candidates}"
        )
    gen_candidates = discover_generator_candidates(
        base_net, base_ppc, args.n_gen_candidates, args.seed
    )
    line_pools = _split_pools(line_candidates, args.seed + 101)
    gen_pools = _split_pools(gen_candidates, args.seed + 211)

    n_train, n_val, n_test = line_ds._counts(args.N)
    split_names = ["train", "val", "test"]
    split_sizes = [n_train, n_val, n_test]
    for name, size in zip(split_names, split_sizes):
        required = 1 + len(line_pools[name]) + len(gen_pools[name])
        if args.ensure_candidate_coverage and size < required:
            raise ValueError(
                f"split={name} has {size} rows but needs at least {required} for clean plus "
                "candidate coverage"
            )

    pd_rows: List[np.ndarray] = []
    pg_rows: List[np.ndarray] = []
    pf_rows: List[np.ndarray] = []
    pt_rows: List[np.ndarray] = []
    line_rows: List[int] = []
    line_pp_rows: List[int] = []
    gen_rows: List[int] = []
    type_rows: List[int] = []
    split_rows: List[int] = []
    balance_rows: List[float] = []
    attempts_by_split: Dict[str, int] = {}
    rejections_by_split: Dict[str, int] = {}
    rejection_reasons: Counter[str] = Counter()
    rejection_scenarios: Counter[str] = Counter()

    for split_id, (name, target) in enumerate(zip(split_names, split_sizes)):
        line_pool = line_pools[name]
        gen_pool = gen_pools[name]
        if not line_pool or not gen_pool:
            raise RuntimeError(f"empty line or generator candidate pool for split={name}")
        mandatory = _mandatory_schedule(line_pool, gen_pool) if args.ensure_candidate_coverage else []
        accepted = 0
        attempts = 0
        max_attempts = max(target, int(np.ceil(args.max_attempt_factor * target)))
        while accepted < target:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"split={name} accepted only {accepted}/{target} after {attempts} attempts"
                )
            attempts += 1
            factors = rng.uniform(args.scale_low, args.scale_high, size=len(base_net.load)).astype(float)
            net = copy.deepcopy(base_net)
            if len(net.load):
                net.load.loc[:, "p_mw"] = net.load.p_mw.values * factors
                if "q_mvar" in net.load.columns:
                    net.load.loc[:, "q_mvar"] = net.load.q_mvar.values * factors

            if accepted < len(mandatory):
                kind, record = mandatory[accepted]
            else:
                kind, record = _choose_random_scenario(
                    rng, probabilities, line_pool, gen_pool
                )

            cont_line_idx = -1
            cont_line_pp_idx = -1
            cont_gen_idx = -1
            if kind == TYPE_LINE:
                assert record is not None
                cont_line_idx = int(record["cont_line_idx"])
                cont_line_pp_idx = int(record["line_pp_idx"])
                net.line.at[cont_line_pp_idx, "in_service"] = False
            elif kind == TYPE_GENERATOR:
                assert record is not None
                cont_gen_idx = int(record["cont_gen_idx"])
                _apply_generator_outage(net, record)

            try:
                sol = line_ds._solve_and_extract(net)
            except OPFNotConverged:
                rejection_reasons["OPFNotConverged"] += 1
                rejection_scenarios[
                    f"{TYPE_NAMES[kind]}:{cont_line_idx if kind == TYPE_LINE else cont_gen_idx}"
                ] += 1
                rejected = attempts - accepted
                if rejected <= 3 or rejected % 25 == 0:
                    print(
                        f"[sample-reject] split={name} attempt={attempts} accepted={accepted}/{target} "
                        f"type={TYPE_NAMES[kind]} cont_line_idx={cont_line_idx} "
                        f"cont_gen_idx={cont_gen_idx} reason=OPFNotConverged",
                        flush=True,
                    )
                continue

            if sol["pd_bus"].shape[0] != static["nbus"] or sol["pg"].shape[0] != static["ngen"]:
                raise RuntimeError("contingency changed bus/generator dimensions")
            if cont_gen_idx >= 0 and abs(float(sol["pg"][cont_gen_idx])) > 1e-3:
                raise RuntimeError("outaged-generator solution is not zero")
            cur_to_base = (
                np.delete(np.arange(len(base_branch), dtype=np.int64), cont_line_idx)
                if cont_line_idx >= 0
                else np.arange(len(base_branch), dtype=np.int64)
            )
            if sol["pf"].shape[0] != cur_to_base.shape[0]:
                raise RuntimeError("post-contingency flow labels do not match base branch ordering")

            pd_rows.append(sol["pd_bus"])
            pg_rows.append(sol["pg"])
            line_rows.append(cont_line_idx)
            line_pp_rows.append(cont_line_pp_idx)
            gen_rows.append(cont_gen_idx)
            type_rows.append(kind)
            split_rows.append(split_id)
            if not args.no_flow_labels:
                pf_base = np.zeros(len(base_branch), dtype=np.float32)
                pt_base = np.zeros(len(base_branch), dtype=np.float32)
                pf_base[cur_to_base] = sol["pf"]
                pt_base[cur_to_base] = sol["pt"]
                pf_rows.append(pf_base)
                pt_rows.append(pt_base)
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
                    f"[sample-progress] global={global_accepted}/{args.N} split={name} "
                    f"split_accepted={accepted}/{target} split_attempts={attempts} "
                    f"split_rejected={attempts - accepted}",
                    flush=True,
                )
        attempts_by_split[name] = attempts
        rejections_by_split[name] = attempts - accepted

    pd_bus = np.stack(pd_rows).astype(np.float32)
    pg = np.stack(pg_rows).astype(np.float32)
    cont_line_idx = np.asarray(line_rows, dtype=np.int64)
    cont_line_pp_idx = np.asarray(line_pp_rows, dtype=np.int64)
    cont_gen_idx = np.asarray(gen_rows, dtype=np.int64)
    contingency_type = np.asarray(type_rows, dtype=np.int8)
    split = np.asarray(split_rows, dtype=np.int8)
    balance = np.asarray(balance_rows, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "pd_bus": pd_bus,
        "pg": pg,
        "cont_line_idx": cont_line_idx,
        "cont_line_pp_idx": cont_line_pp_idx,
        "cont_gen_idx": cont_gen_idx,
        "contingency_type": contingency_type,
        "split": split,
        "gs_bus": np.asarray(static["gs_bus"], dtype=np.float32),
        "pd_bus_includes_gs": np.int8(0),
        "dataset_schema_version": np.int16(2),
        "dataset_schema_name": np.asarray("fair4net_mixed_n1_contingency"),
        "case_name": np.asarray(shared.CASE_NAME),
        "demand_convention": np.asarray("effective_demand=pd_bus+gs_bus"),
        "gs_total_mw": np.float32(np.sum(static["gs_bus"])),
        "lodf": lodf_columns,
    }
    if not args.no_flow_labels:
        payload["pf"] = np.stack(pf_rows).astype(np.float32)
        payload["pt"] = np.stack(pt_rows).astype(np.float32)
    np.savez_compressed(args.out, **payload)

    meta_path = args.meta if str(args.meta) else args.out.with_suffix(".meta.json")
    category_counts = {
        name: {
            TYPE_NAMES[kind]: int(np.count_nonzero((split == split_id) & (contingency_type == kind)))
            for kind in TYPE_NAMES
        }
        for split_id, name in enumerate(split_names)
    }
    meta = {
        "case_name": shared.CASE_NAME,
        "dataset_schema_name": "fair4net_mixed_n1_contingency",
        "dataset_schema_version": 2,
        "N": int(args.N),
        "seed": int(args.seed),
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "category_probabilities": {
            "clean": float(args.p_clean),
            "line": float(args.p_line),
            "generator": float(args.p_gen),
        },
        "category_counts": category_counts,
        "ensure_candidate_coverage": bool(args.ensure_candidate_coverage),
        "progress_every": int(args.progress_every),
        "line_index_contract": "base internal PYPOWER branch row; -1 means no line outage",
        "generator_index_contract": "base internal PYPOWER generator row; -1 means no generator outage",
        "scenario_contract": "exactly one of clean, single-line outage, single-generator outage; never simultaneous",
        "split_contract": "line and generator outage identities are each disjoint across train/val/test",
        "flow_model_contract": tr.FLOW_MODEL_CONTRACT,
        "demand_convention": "pd_bus is conventional load; effective demand=pd_bus+gs_bus",
        "gs_total_mw": float(np.sum(static["gs_bus"])),
        "line_candidate_pools": line_pools,
        "generator_candidate_pools": gen_pools,
        "base_nbranch": int(static["nbranch"]),
        "base_nbus": int(static["nbus"]),
        "base_ngen": int(static["ngen"]),
        "balance_abs_mean_MW": float(np.mean(np.abs(balance))),
        "balance_abs_max_MW": float(np.max(np.abs(balance))),
        "balance_abs_p95_MW": float(np.percentile(np.abs(balance), 95)),
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
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "rejections_by_scenario": dict(sorted(rejection_scenarios.items())),
        },
        "notes": (
            "Line scenarios use base PTDF plus an audited LODF column. Generator scenarios "
            "retain base topology and force one internal generator bound to [0,0]."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "meta": str(meta_path),
                "case_name": shared.CASE_NAME,
                "N": int(args.N),
                "split_counts": meta["split_counts"],
                "category_counts": category_counts,
                "balance_abs_max_MW": meta["balance_abs_max_MW"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
