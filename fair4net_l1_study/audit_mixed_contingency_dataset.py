#!/usr/bin/env python3
"""Fail-fast physics and split audit for mixed N-1 contingency datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import torch

import dc_opf_pandapower_case300_9k as shared
import deq_trainer_fair4net as tr
from make_mixed_contingency_dataset import TYPE_CLEAN, TYPE_GENERATOR, TYPE_LINE


SPLIT_NAMES = ("train", "val", "test")
TYPE_NAMES = {TYPE_CLEAN: "clean", TYPE_LINE: "line", TYPE_GENERATOR: "generator"}


def _pool_ids(
    meta: Dict[str, Any], key: str, id_key: str
) -> List[Set[int]]:
    pools = meta.get(key, {})
    return [set(int(row[id_key]) for row in pools.get(name, [])) for name in SPLIT_NAMES]


def _disjoint(label: str, sets: Iterable[Set[int]]) -> None:
    values = list(sets)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            overlap = values[i] & values[j]
            if overlap:
                raise ValueError(f"{label} overlap across splits: {sorted(overlap)}")


def _row_digest(row: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(row).view(np.uint8), digest_size=16).digest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--expected-N", type=int)
    ap.add_argument("--require-candidate-coverage", action="store_true")
    ap.add_argument("--balance-tol", type=float, default=1e-2)
    ap.add_argument("--flow-reconstruction-tol", type=float, default=1e-2)
    ap.add_argument("--operator-flow-reconstruction-atol", type=float, default=5e-2)
    ap.add_argument("--operator-flow-reconstruction-rtol", type=float, default=5e-5)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("case_name") != shared.CASE_NAME:
        raise ValueError(
            f"dataset case {meta.get('case_name')!r} does not match DEQ_CASE={shared.CASE_NAME!r}"
        )
    if meta.get("dataset_schema_name") != "fair4net_mixed_n1_contingency":
        raise ValueError("unexpected mixed-contingency dataset schema")
    if int(meta.get("dataset_schema_version", -1)) != 2:
        raise ValueError("unexpected mixed-contingency schema version")

    required = {
        "pd_bus", "pg", "pf", "pt", "cont_line_idx", "cont_line_pp_idx",
        "cont_gen_idx", "contingency_type", "split", "gs_bus",
        "pd_bus_includes_gs", "lodf",
    }
    with np.load(args.data, allow_pickle=False) as z:
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"missing dataset fields: {missing}")
        pd_bus = z["pd_bus"].astype(np.float32)
        pg = z["pg"].astype(np.float32)
        pf = z["pf"].astype(np.float32)
        pt = z["pt"].astype(np.float32)
        cont_line = z["cont_line_idx"].astype(np.int64)
        cont_line_pp = z["cont_line_pp_idx"].astype(np.int64)
        cont_gen = z["cont_gen_idx"].astype(np.int64)
        kind = z["contingency_type"].astype(np.int64)
        split = z["split"].astype(np.int64)
        gs = z["gs_bus"].astype(np.float32)
        lodf = z["lodf"].astype(np.float32)
        pd_includes_gs = int(np.asarray(z["pd_bus_includes_gs"]).item())

    n = int(pd_bus.shape[0])
    if args.expected_N is not None and n != args.expected_N:
        raise ValueError(f"dataset N={n} does not match expected N={args.expected_N}")
    if pd_bus.ndim != 2 or pg.ndim != 2:
        raise ValueError(f"bad pd/pg shapes: {pd_bus.shape} {pg.shape}")
    if not np.all(np.isfinite(pd_bus)) or not np.all(np.isfinite(pg)):
        raise ValueError("pd_bus or pg contains non-finite values")
    for label, value in {
        "cont_line_idx": cont_line,
        "cont_line_pp_idx": cont_line_pp,
        "cont_gen_idx": cont_gen,
        "contingency_type": kind,
        "split": split,
    }.items():
        if value.shape != (n,):
            raise ValueError(f"{label} shape {value.shape} does not match N={n}")
    if pf.shape != pt.shape or pf.shape[0] != n:
        raise ValueError("pf/pt shapes do not match N")
    if pd_includes_gs != 0:
        raise ValueError("pd_bus_includes_gs must be zero")

    base_ppc = tr.build_ppc_from_net(tr.build_base_net())
    static = tr.extract_static_mats(base_ppc)
    nbus = int(static["nbus"])
    ngen = int(static["ngen"])
    nbranch = int(static["nbranch"])
    if pd_bus.shape != (n, nbus) or pg.shape != (n, ngen) or pf.shape != (n, nbranch):
        raise ValueError(
            f"network shapes differ: pd={pd_bus.shape}, pg={pg.shape}, flow={pf.shape}, "
            f"expected=({n},{nbus})/({n},{ngen})/({n},{nbranch})"
        )
    if gs.shape != (nbus,) or not np.allclose(gs, static["gs_bus"], atol=1e-6, rtol=0.0):
        raise ValueError("dataset gs_bus does not match the selected network")
    if lodf.shape != (nbranch, nbranch) or not np.all(np.isfinite(lodf)):
        raise ValueError("embedded LODF matrix has the wrong shape or non-finite entries")
    if np.any((cont_line < -1) | (cont_line >= nbranch)):
        raise ValueError("cont_line_idx is out of range")
    if np.any((cont_gen < -1) | (cont_gen >= ngen)):
        raise ValueError("cont_gen_idx is out of range")
    if np.any((cont_line < 0) != (cont_line_pp < 0)):
        raise ValueError("line internal and pandapower indices disagree on no-outage rows")
    if set(np.unique(split).tolist()) != {0, 1, 2}:
        raise ValueError("split must contain exactly train=0, val=1, test=2")
    if not set(np.unique(kind).tolist()).issubset(TYPE_NAMES):
        raise ValueError(f"unexpected contingency type values: {np.unique(kind).tolist()}")

    clean_mask = kind == TYPE_CLEAN
    line_mask = kind == TYPE_LINE
    gen_mask = kind == TYPE_GENERATOR
    if not np.all(clean_mask == ((cont_line < 0) & (cont_gen < 0))):
        raise ValueError("clean rows do not have both outage indices equal to -1")
    if not np.all(line_mask == ((cont_line >= 0) & (cont_gen < 0))):
        raise ValueError("line rows violate the exclusive line-outage contract")
    if not np.all(gen_mask == ((cont_gen >= 0) & (cont_line < 0))):
        raise ValueError("generator rows violate the exclusive generator-outage contract")
    if np.any((cont_line >= 0) & (cont_gen >= 0)):
        raise ValueError("simultaneous line+generator outages are not allowed in this N-1 study")

    unique, counts = np.unique(split, return_counts=True)
    split_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    category_counts: Dict[str, Dict[str, int]] = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        category_counts[split_name] = {
            type_name: int(np.count_nonzero((split == split_id) & (kind == type_id)))
            for type_id, type_name in TYPE_NAMES.items()
        }
        if any(value <= 0 for value in category_counts[split_name].values()):
            raise ValueError(f"split={split_name} does not contain clean, line, and generator rows")

    # Demand vectors are independently sampled.  Exact cross-split duplicates
    # would violate the requested train/test sample isolation even when outage
    # IDs are disjoint.
    demand_digests = [
        {_row_digest(row) for row in pd_bus[split == split_id]}
        for split_id in range(3)
    ]
    _disjoint("exact demand-scenario digest", demand_digests)

    line_pools = _pool_ids(meta, "line_candidate_pools", "cont_line_idx")
    gen_pools = _pool_ids(meta, "generator_candidate_pools", "cont_gen_idx")
    _disjoint("declared line candidate", line_pools)
    _disjoint("declared generator candidate", gen_pools)
    sampled_line = [set(int(x) for x in cont_line[(split == i) & line_mask]) for i in range(3)]
    sampled_gen = [set(int(x) for x in cont_gen[(split == i) & gen_mask]) for i in range(3)]
    _disjoint("sampled line outage", sampled_line)
    _disjoint("sampled generator outage", sampled_gen)
    for i, name in enumerate(SPLIT_NAMES):
        if not sampled_line[i].issubset(line_pools[i]):
            raise ValueError(f"{name} contains a line outage outside its declared pool")
        if not sampled_gen[i].issubset(gen_pools[i]):
            raise ValueError(f"{name} contains a generator outage outside its declared pool")
    if args.require_candidate_coverage:
        if not bool(meta.get("ensure_candidate_coverage", False)):
            raise ValueError("metadata does not attest candidate coverage")
        for i, name in enumerate(SPLIT_NAMES):
            if sampled_line[i] != line_pools[i]:
                raise ValueError(f"{name} line candidate coverage is incomplete")
            if sampled_gen[i] != gen_pools[i]:
                raise ValueError(f"{name} generator candidate coverage is incomplete")

    line_candidates = set().union(*line_pools)
    gen_candidates = set().union(*gen_pools)
    if any(np.allclose(lodf[:, col], 0.0) for col in line_candidates):
        raise ValueError("a declared line candidate has a zero LODF column")
    if any(not np.isclose(lodf[col, col], -1.0, atol=1e-6, rtol=0.0) for col in line_candidates):
        raise ValueError("a declared line candidate violates LODF[k,k]=-1")

    ptdf = np.asarray(static["ptdf_dense"], dtype=np.float64)
    ptdf_delta: List[float] = []
    outaged_ptdf_row: List[float] = []
    for col in sorted(line_candidates):
        scenario = ptdf + lodf[:, col, None] * ptdf[col, None, :]
        ptdf_delta.append(float(np.max(np.abs(scenario - ptdf))))
        outaged_ptdf_row.append(float(np.max(np.abs(scenario[col]))))
    if ptdf_delta and min(ptdf_delta) <= 1e-8:
        raise ValueError("a declared line outage does not materially change the PTDF")
    if outaged_ptdf_row and max(outaged_ptdf_row) > 1e-6:
        raise ValueError("a post-outage PTDF does not zero the outaged branch row")

    balance = (
        pg.astype(np.float64).sum(axis=1)
        - pd_bus.astype(np.float64).sum(axis=1)
        - gs.astype(np.float64).sum()
    )
    max_balance = float(np.max(np.abs(balance)))
    if max_balance > args.balance_tol:
        raise ValueError(f"balance audit failed: {max_balance:.6g} MW > {args.balance_tol}")

    outaged_gen_max = 0.0
    if np.any(gen_mask):
        rows = np.flatnonzero(gen_mask)
        outaged_gen_max = float(np.max(np.abs(pg[rows, cont_gen[rows]])))
        if outaged_gen_max > 1e-3:
            raise ValueError(f"outaged generator labels are not zero: {outaged_gen_max:.6g} MW")

    outaged_line_flow_max = 0.0
    if np.any(line_mask):
        rows = np.flatnonzero(line_mask)
        outaged_line_flow_max = float(
            max(
                np.max(np.abs(pf[rows, cont_line[rows]])),
                np.max(np.abs(pt[rows, cont_line[rows]])),
            )
        )
        if outaged_line_flow_max > 1e-3:
            raise ValueError(f"outaged branch labels are not zero: {outaged_line_flow_max:.6g} MW")

    inj = np.zeros((n, nbus), dtype=np.float64)
    for gen_idx, bus_idx in enumerate(np.asarray(static["bus_of_gen"], dtype=np.int64)):
        inj[:, bus_idx] += pg[:, gen_idx]
    inj -= pd_bus.astype(np.float64) + gs.astype(np.float64)[None, :]
    base_flow = inj @ ptdf.T + np.asarray(static["flow_offset"], dtype=np.float64)[None, :]
    safe_line = np.clip(cont_line, 0, nbranch - 1)
    selected_lodf = lodf[:, safe_line].T.astype(np.float64)
    selected_lodf[cont_line < 0] = 0.0
    reconstructed = base_flow + selected_lodf * base_flow[np.arange(n), safe_line, None]
    flow_reconstruction_max = float(
        max(
            np.max(np.abs(reconstructed - pf.astype(np.float64))),
            np.max(np.abs(reconstructed + pt.astype(np.float64))),
        )
    )
    if flow_reconstruction_max > args.flow_reconstruction_tol:
        raise ValueError(
            f"PTDF+LODF reconstruction failed: {flow_reconstruction_max:.6g} MW "
            f"> {args.flow_reconstruction_tol}"
        )

    static_for_op = dict(static)
    static_for_op["lodf"] = lodf
    cfg = tr.TrainCfg(device="cpu")
    torch_static = tr.to_torch_static(static_for_op, cfg)
    steps = tr.LearnableSteps(nK=1.0, s_max=cfg.s_max)
    op = tr.PDHGOperatorLearnable(torch_static, steps)
    with torch.no_grad():
        operator_flow = op.branch_flows(
            torch.as_tensor(pg, dtype=cfg.dtype),
            torch.as_tensor(pd_bus, dtype=cfg.dtype),
            torch.as_tensor(cont_line, dtype=torch.long),
        ).cpu().numpy()
    error_pf = np.abs(operator_flow - pf)
    error_pt = np.abs(operator_flow + pt)
    operator_max = float(max(np.max(error_pf), np.max(error_pt)))
    tol_pf = args.operator_flow_reconstruction_atol + args.operator_flow_reconstruction_rtol * np.abs(pf)
    tol_pt = args.operator_flow_reconstruction_atol + args.operator_flow_reconstruction_rtol * np.abs(pt)
    operator_scaled = float(max(np.max(error_pf / tol_pf), np.max(error_pt / tol_pt)))
    if operator_scaled > 1.0:
        raise ValueError(
            f"PyTorch operator reconstruction failed: max={operator_max:.6g} MW, "
            f"scaled_max={operator_scaled:.6g}"
        )

    sampling = meta.get("opf_sampling", {})
    accepted = int(sampling.get("accepted_total", -1))
    attempts = int(sampling.get("attempts_total", -1))
    rejected = int(sampling.get("rejections_total", -1))
    if sampling.get("policy") != "reject_and_resample_OPFNotConverged_only":
        raise ValueError("unrecognized OPF sampling policy")
    if accepted != n or attempts != accepted + rejected:
        raise ValueError("OPF attempt/rejection accounting is inconsistent")

    report = {
        "status": "PASS",
        "protocol": "fair4net_mixed_n1_contingency_v1",
        "case": shared.CASE_NAME,
        "N": n,
        "shape": {"nbus": nbus, "ngen": ngen, "nbranch": nbranch},
        "split_counts": split_counts,
        "category_counts": category_counts,
        "clean_samples": int(np.count_nonzero(clean_mask)),
        "line_outage_samples": int(np.count_nonzero(line_mask)),
        "generator_outage_samples": int(np.count_nonzero(gen_mask)),
        "line_candidates": int(len(line_candidates)),
        "generator_candidates": int(len(gen_candidates)),
        "train_line_outage_ids": sorted(sampled_line[0]),
        "validation_line_outage_ids": sorted(sampled_line[1]),
        "test_line_outage_ids": sorted(sampled_line[2]),
        "train_generator_outage_ids": sorted(sampled_gen[0]),
        "validation_generator_outage_ids": sorted(sampled_gen[1]),
        "test_generator_outage_ids": sorted(sampled_gen[2]),
        "cross_split_exact_demand_duplicates": 0,
        "simultaneous_line_generator_samples": 0,
        "flow_labels_saved": True,
        "ptdf_change_abs_min": min(ptdf_delta) if ptdf_delta else 0.0,
        "outaged_ptdf_row_abs_max": max(outaged_ptdf_row) if outaged_ptdf_row else 0.0,
        "flow_reconstruction_abs_max_MW": flow_reconstruction_max,
        "operator_flow_reconstruction_abs_max_MW": operator_max,
        "operator_flow_reconstruction_scaled_max": operator_scaled,
        "operator_flow_reconstruction_atol_MW": float(args.operator_flow_reconstruction_atol),
        "operator_flow_reconstruction_rtol": float(args.operator_flow_reconstruction_rtol),
        "balance_abs_mean_MW": float(np.mean(np.abs(balance))),
        "balance_abs_max_MW": max_balance,
        "outaged_generator_label_abs_max_MW": outaged_gen_max,
        "outaged_line_flow_label_abs_max_MW": outaged_line_flow_max,
        "opf_sampling": sampling,
        "split_isolation": {
            "line_outage_id_overlap": 0,
            "generator_outage_id_overlap": 0,
            "exact_demand_row_overlap": 0,
        },
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
