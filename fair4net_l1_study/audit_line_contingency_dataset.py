#!/usr/bin/env python3
"""Fail-fast audit for a GS-aware disjoint line-contingency dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dc_opf_pandapower_case300_9k as shared
import deq_trainer_fair4net as tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="optional persistent JSON audit report")
    ap.add_argument("--expected-N", type=int, help="require this exact number of accepted scenarios")
    ap.add_argument(
        "--require-candidate-coverage",
        action="store_true",
        help="require every declared split-specific outage candidate to occur in the saved rows",
    )
    # pandapower's solved case300 labels show sub-0.01 MW numerical balance
    # noise under some N-1 topologies (observed max 0.00665 MW locally).
    ap.add_argument("--balance-tol", type=float, default=1e-2)
    ap.add_argument("--flow-reconstruction-tol", type=float, default=1e-2)
    ap.add_argument(
        "--operator-flow-reconstruction-atol",
        type=float,
        default=5e-2,
        help=(
            "absolute tolerance for the production float32 PyTorch operator; "
            "the independent float64 physics reconstruction retains the "
            "stricter --flow-reconstruction-tol gate"
        ),
    )
    ap.add_argument(
        "--operator-flow-reconstruction-rtol",
        type=float,
        default=5e-5,
        help=(
            "relative tolerance for the float32 PyTorch operator check; the "
            "independent NumPy physics reconstruction retains the strict "
            "absolute --flow-reconstruction-tol gate"
        ),
    )
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("case_name") != shared.CASE_NAME:
        raise ValueError(
            f"dataset case {meta.get('case_name')!r} does not match DEQ_CASE={shared.CASE_NAME!r}"
        )
    with np.load(args.data, allow_pickle=False) as z:
        required = {"pd_bus", "pg", "cont_line_idx", "cont_line_pp_idx", "split", "gs_bus", "pd_bus_includes_gs", "lodf"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"missing dataset fields: {missing}")
        pd_bus = z["pd_bus"].astype(np.float32)
        pg = z["pg"].astype(np.float32)
        pf = z["pf"].astype(np.float32) if "pf" in z.files else None
        pt = z["pt"].astype(np.float32) if "pt" in z.files else None
        cont = z["cont_line_idx"].astype(np.int64)
        cont_pp = z["cont_line_pp_idx"].astype(np.int64)
        split = z["split"].astype(np.int64)
        gs = z["gs_bus"].astype(np.float32)
        lodf = z["lodf"].astype(np.float32)
        if int(np.asarray(z["pd_bus_includes_gs"]).item()) != 0:
            raise ValueError("pd_bus_includes_gs must be 0")

    if pd_bus.ndim != 2 or pg.ndim != 2 or pd_bus.shape[0] != pg.shape[0]:
        raise ValueError(f"bad pd/pg shapes: {pd_bus.shape} {pg.shape}")
    if args.expected_N is not None and pd_bus.shape[0] != args.expected_N:
        raise ValueError(f"dataset N={pd_bus.shape[0]} does not match expected N={args.expected_N}")
    if (pf is None) != (pt is None):
        raise ValueError("pf and pt must be saved together")
    if pf is not None and (pf.shape[0] != pd_bus.shape[0] or pt.shape != pf.shape):
        raise ValueError("flow-label shapes do not match N")
    if gs.shape != (pd_bus.shape[1],):
        raise ValueError(f"gs_bus shape {gs.shape} != ({pd_bus.shape[1]},)")
    expected_lodf = (int(meta["base_nbranch"]), int(meta["base_nbranch"]))
    if lodf.shape != expected_lodf or not np.all(np.isfinite(lodf)):
        raise ValueError(f"lodf shape/finite check failed: {lodf.shape} expected={expected_lodf}")
    base_ppc = tr.build_ppc_from_net(tr.build_base_net())
    base_static = tr.extract_static_mats(base_ppc)
    if not np.allclose(gs, base_static["gs_bus"], atol=1e-6, rtol=0.0):
        raise ValueError("dataset gs_bus does not match the selected network")

    unique, counts = np.unique(split, return_counts=True)
    if set(unique.tolist()) != {0, 1, 2}:
        raise ValueError(f"split must contain exactly 0,1,2; got {unique.tolist()}")
    if np.any((cont < -1) | (cont >= int(meta["base_nbranch"]))):
        raise ValueError("cont_line_idx is out of range")
    if np.any((cont < 0) != (cont_pp < 0)):
        raise ValueError("cont_line_idx and cont_line_pp_idx disagree on no-outage rows")

    # Use float64 reductions over the stored float32 labels.  Float32
    # accumulation alone can exceed 0.01 MW on thousand-bus cases even when
    # the independently solved OPF labels satisfy balance.
    balance = (
        pg.astype(np.float64).sum(axis=1)
        - pd_bus.astype(np.float64).sum(axis=1)
        - gs.astype(np.float64).sum()
    )
    max_balance = float(np.max(np.abs(balance)))
    if max_balance > args.balance_tol:
        raise ValueError(f"balance audit failed: max={max_balance:.6g} MW > {args.balance_tol}")

    opf_sampling = meta.get("opf_sampling", {})
    if opf_sampling:
        accepted_total = int(opf_sampling.get("accepted_total", -1))
        attempts_total = int(opf_sampling.get("attempts_total", -1))
        rejections_total = int(opf_sampling.get("rejections_total", -1))
        if accepted_total != int(pd_bus.shape[0]):
            raise ValueError("OPF sampling accepted_total does not match dataset N")
        if attempts_total != accepted_total + rejections_total:
            raise ValueError("OPF sampling attempts/rejections accounting is inconsistent")
        if opf_sampling.get("policy") != "reject_and_resample_OPFNotConverged_only":
            raise ValueError("unrecognized OPF sampling/rejection policy")
    outaged_flow_max = 0.0
    if pf is not None:
        rows = np.flatnonzero(cont >= 0)
        if len(rows):
            outaged_flow_max = float(max(
                np.max(np.abs(pf[rows, cont[rows]])),
                np.max(np.abs(pt[rows, cont[rows]])),
            ))
            if outaged_flow_max > 1e-3:
                raise ValueError(f"outaged branch labels are not zero: max={outaged_flow_max:.6g} MW")

    pools = meta.get("line_candidate_pools", {})
    sets = [set(int(x["cont_line_idx"]) for x in pools.get(k, [])) for k in ("train", "val", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("line-contingency pools overlap across splits")
    sample_sets = [set(int(x) for x in cont[(split == i) & (cont >= 0)]) for i in range(3)]
    if sample_sets[0] & sample_sets[1] or sample_sets[0] & sample_sets[2] or sample_sets[1] & sample_sets[2]:
        raise ValueError("sampled line contingencies overlap across train/validation/test")
    for i, name in enumerate(("train", "validation", "test")):
        if not sample_sets[i].issubset(sets[i]):
            raise ValueError(f"{name} samples contain an outage outside their declared pool")
    if args.require_candidate_coverage:
        if not bool(meta.get("ensure_candidate_coverage", False)):
            raise ValueError("metadata does not attest ensure_candidate_coverage=true")
        for i, name in enumerate(("train", "validation", "test")):
            if sample_sets[i] != sets[i]:
                missing = sorted(sets[i] - sample_sets[i])
                raise ValueError(f"{name} is missing declared outage candidates: {missing}")
    candidate_cols = set().union(*sets)
    if any(np.allclose(lodf[:, col], 0.0) for col in candidate_cols):
        raise ValueError("one or more declared outage candidates has a zero LODF column")
    if any(not np.isclose(lodf[col, col], -1.0, atol=1e-6, rtol=0.0) for col in candidate_cols):
        raise ValueError("one or more outage candidates violates LODF[k,k] = -1")

    # Verify every declared contingency actually changes the PTDF and zeros the
    # outaged branch row under the compact rank-one reconstruction.
    ptdf = np.asarray(base_static["ptdf_dense"], dtype=np.float64)
    ptdf_delta_max = []
    outaged_ptdf_row_max = []
    for col in sorted(candidate_cols):
        scenario_ptdf = ptdf + lodf[:, col, None] * ptdf[col, None, :]
        ptdf_delta_max.append(float(np.max(np.abs(scenario_ptdf - ptdf))))
        outaged_ptdf_row_max.append(float(np.max(np.abs(scenario_ptdf[col]))))
    if ptdf_delta_max and min(ptdf_delta_max) <= 1e-8:
        raise ValueError("one or more declared outages does not change the PTDF")
    if outaged_ptdf_row_max and max(outaged_ptdf_row_max) > 1e-6:
        raise ValueError("post-outage PTDF does not zero the outaged branch row")

    # End-to-end physics check: reconstruct every labeled branch flow from the
    # base PTDF, phase-shift offset, and selected LODF column, then compare with
    # the independently solved post-outage labels saved by pandapower.
    flow_reconstruction_max = 0.0
    operator_flow_reconstruction_max = 0.0
    operator_flow_reconstruction_scaled_max = 0.0
    if pf is not None:
        inj = np.zeros((pd_bus.shape[0], int(base_static["nbus"])), dtype=np.float64)
        for gen_idx, bus_idx in enumerate(np.asarray(base_static["bus_of_gen"], dtype=np.int64)):
            inj[:, bus_idx] += pg[:, gen_idx]
        inj -= pd_bus.astype(np.float64) + gs.astype(np.float64)[None, :]
        base_flow = inj @ ptdf.T + np.asarray(base_static["flow_offset"], dtype=np.float64)[None, :]
        safe = np.clip(cont, 0, int(base_static["nbranch"]) - 1)
        selected_lodf = lodf[:, safe].T.astype(np.float64)
        selected_lodf[cont < 0] = 0.0
        reconstructed = base_flow + selected_lodf * base_flow[np.arange(pd_bus.shape[0]), safe, None]
        flow_reconstruction_max = float(max(
            np.max(np.abs(reconstructed - pf.astype(np.float64))),
            np.max(np.abs(reconstructed + pt.astype(np.float64))),
        ))
        if flow_reconstruction_max > args.flow_reconstruction_tol:
            raise ValueError(
                f"PTDF+LODF flow reconstruction failed: max={flow_reconstruction_max:.6g} MW "
                f"> {args.flow_reconstruction_tol}"
            )

        # Exercise the exact PyTorch operator used by both DEQ and baselines;
        # this catches accidental row-vs-column LODF indexing regressions.
        static_for_op = dict(base_static)
        static_for_op["lodf"] = lodf
        cfg = tr.TrainCfg(device="cpu")
        torch_static = tr.to_torch_static(static_for_op, cfg)
        steps = tr.LearnableSteps(nK=1.0, s_max=cfg.s_max)
        op = tr.PDHGOperatorLearnable(torch_static, steps)
        with torch.no_grad():
            operator_flow = op.branch_flows(
                torch.as_tensor(pg, dtype=cfg.dtype),
                torch.as_tensor(pd_bus, dtype=cfg.dtype),
                torch.as_tensor(cont, dtype=torch.long),
            ).cpu().numpy()
        operator_error_pf = np.abs(operator_flow - pf)
        operator_error_pt = np.abs(operator_flow + pt)
        operator_flow_reconstruction_max = float(max(
            np.max(operator_error_pf),
            np.max(operator_error_pt),
        ))
        # The production operator intentionally runs in float32.  A fixed
        # absolute cutoff is not portable across BLAS backends for thousand-bus
        # matrix products, so test it with an elementwise atol+rtol bound.  This
        # remains tight near zero while allowing expected relative roundoff on
        # multi-thousand-MW flows.  The independent float64 reconstruction above
        # still has the strict absolute physics gate.
        operator_tol_pf = (
            args.operator_flow_reconstruction_atol
            + args.operator_flow_reconstruction_rtol * np.abs(pf)
        )
        operator_tol_pt = (
            args.operator_flow_reconstruction_atol
            + args.operator_flow_reconstruction_rtol * np.abs(pt)
        )
        operator_flow_reconstruction_scaled_max = float(max(
            np.max(operator_error_pf / operator_tol_pf),
            np.max(operator_error_pt / operator_tol_pt),
        ))
        if operator_flow_reconstruction_scaled_max > 1.0:
            raise ValueError(
                f"PyTorch contingency operator flow reconstruction failed: "
                f"max={operator_flow_reconstruction_max:.6g} MW, "
                f"scaled_max={operator_flow_reconstruction_scaled_max:.6g} > 1 "
                f"(atol={args.operator_flow_reconstruction_atol}, "
                f"rtol={args.operator_flow_reconstruction_rtol})"
            )
    split_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    report = {
        "status": "PASS",
        "case": shared.CASE_NAME,
        "N": int(pd_bus.shape[0]),
        "shape": {"nbus": int(pd_bus.shape[1]), "ngen": int(pg.shape[1]), "nbranch": int(meta["base_nbranch"])},
        "flow_labels_saved": pf is not None,
        "lodf_candidates": int(len(candidate_cols)),
        "split_counts": split_counts,
        "outage_samples": int(np.count_nonzero(cont >= 0)),
        "unique_outages": int(len(np.unique(cont[cont >= 0]))),
        "opf_sampling": opf_sampling,
        "train_outage_ids": sorted(sample_sets[0]),
        "validation_outage_ids": sorted(sample_sets[1]),
        "test_outage_ids": sorted(sample_sets[2]),
        "ptdf_change_abs_min": min(ptdf_delta_max) if ptdf_delta_max else 0.0,
        "outaged_ptdf_row_abs_max": max(outaged_ptdf_row_max) if outaged_ptdf_row_max else 0.0,
        "flow_reconstruction_abs_max_MW": flow_reconstruction_max,
        "operator_flow_reconstruction_abs_max_MW": operator_flow_reconstruction_max,
        "operator_flow_reconstruction_scaled_max": operator_flow_reconstruction_scaled_max,
        "operator_flow_reconstruction_atol_MW": float(args.operator_flow_reconstruction_atol),
        "operator_flow_reconstruction_rtol": float(args.operator_flow_reconstruction_rtol),
        "gs_total_mw": float(gs.sum()),
        "balance_abs_mean_MW": float(np.mean(np.abs(balance))),
        "balance_abs_max_MW": max_balance,
        "outaged_flow_label_abs_max_MW": outaged_flow_max,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
