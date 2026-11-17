#!/usr/bin/env python3
"""Add an explicit GS contract to one labeled dataset and audit its labels.

The existing OPF labels already include fixed active shunt demand. This script
does not alter ``pd_bus`` or ``pg``; it adds the static ``gs_bus`` vector and
contract metadata to a new NPZ, then evaluates the solver labels with the same
operator used by DEQ and the corrected baselines.  That operator includes the
constant branch-flow term induced by transformer phase shifters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np


EXPECTED = {
    "case300": (300, 69, 411),
    "case1354pegase": (1354, 260, 1991),
    "case2869pegase": (2869, 510, 4582),
    "case9241pegase": (9241, 1445, 16049),
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.view(np.uint8))
    return h.hexdigest()


def json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def make_corrected_copy(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    case: str,
    gs_bus: np.ndarray,
    line_limit_scale: float,
) -> None:
    if source.resolve() == output.resolve():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    with np.load(source, allow_pickle=True) as z:
        payload = {name: np.array(z[name], copy=True) for name in z.files}
    payload.update(
        gs_bus=np.asarray(gs_bus, dtype=np.float32),
        pd_bus_includes_gs=np.asarray(0, dtype=np.int8),
        contract_version=np.asarray("fair4net_l1_v1"),
        case_name=np.asarray(case),
        source_sha256=np.asarray(source_hash),
        line_limit_scale=np.asarray(line_limit_scale, dtype=np.float64),
        line_limit_convention=np.asarray(
            "legacy benchmark convention retained; line ampacity and loading-percent both scaled"
        ),
    )
    tmp = output.with_name(output.stem + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    # Uncompressed output is intentional: it avoids spending billed node time
    # recompressing hundreds of MB while preserving every source array exactly.
    np.savez(tmp, **payload)
    tmp.replace(output)


def audit_flow_labels(op, d: dict, device: str, batch_size: int) -> dict[str, float]:
    import torch

    test = np.where(np.asarray(d["split"]).astype(np.int64) == 2)[0]
    if test.size == 0:
        raise ValueError("Dataset has no test samples.")
    worst_sum = 0.0
    branch_sum = 0.0
    branch_n = 0
    global_max = 0.0
    sat_n = 0
    n = 0
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            idx = test[start : start + batch_size]
            pd = torch.from_numpy(d["pd_bus"][idx]).to(device=device, dtype=torch.float32)
            pg = torch.from_numpy(d["pg"][idx]).to(device=device, dtype=torch.float32)
            matrix = op.flow_violation_vec(pg, pd)
            worst = matrix.max(dim=1).values
            worst_sum += float(worst.sum().item())
            branch_sum += float(matrix.sum().item())
            branch_n += int(matrix.numel())
            global_max = max(global_max, float(matrix.max().item()))
            sat_n += int((worst <= 1e-3).sum().item())
            n += int(worst.numel())
    return {
        "test_samples": n,
        "label_viol_worst_mean_MW": worst_sum / max(1, n),
        "label_viol_branch_mean_MW": branch_sum / max(1, branch_n),
        "label_viol_max_MW": global_max,
        "label_flow_sat_rate_1e-3": sat_n / max(1, n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(EXPECTED))
    ap.add_argument("--input", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--audit-out", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    os.environ["DEQ_CASE"] = args.case

    # Import only after DEQ_CASE is fixed; the shared case module reads it once.
    import pandapower
    import torch
    import baseline_harness_fair4net as H
    import deq_trainer_fair4net as trainer

    op, static, _ = H.build_operator(args.device)
    dims = (int(static["nbus"]), int(static["ngen"]), int(static["nbranch"]))
    if dims != EXPECTED[args.case]:
        raise ValueError(f"Unexpected network dimensions for {args.case}: {dims}")
    gs_bus = np.asarray(static["gs_bus"], dtype=np.float32)

    if args.output.exists() and not args.force:
        print(f"[dataset] reuse existing corrected artifact: {args.output}")
    else:
        make_corrected_copy(
            args.input,
            args.output,
            case=args.case,
            gs_bus=gs_bus,
            line_limit_scale=float(trainer.LINE_LIMIT_SCALE),
        )
        print(f"[dataset] wrote corrected artifact: {args.output}")

    d = H.load_gencont(str(args.output))
    H.validate_contract(d, static, require_metadata=True)
    if d["pg"] is None:
        raise ValueError("Fair-study dataset must contain OPF pg labels.")
    if d["pd_bus"].shape[0] != 10000:
        raise ValueError(f"Expected N=10000, got {d['pd_bus'].shape[0]}.")

    split = np.asarray(d["split"]).astype(np.int64)
    split_counts = {str(k): int(np.sum(split == k)) for k in (0, 1, 2)}
    if split_counts != {"0": 7000, "1": 1000, "2": 2000}:
        raise ValueError(f"Expected split counts 7000/1000/2000, got {split_counts}.")

    d_eff = d["pd_bus"] + gs_bus[None, :]
    corrected_balance = d["pg"].sum(axis=1, dtype=np.float64) - d_eff.sum(axis=1, dtype=np.float64)
    raw_balance = d["pg"].sum(axis=1, dtype=np.float64) - d["pd_bus"].sum(axis=1, dtype=np.float64)
    bal_mean = float(np.mean(np.abs(corrected_balance)))
    bal_max = float(np.max(np.abs(corrected_balance)))
    if bal_mean > 0.01 or bal_max > 0.05:
        raise ValueError(
            f"Corrected labels fail balance audit: mean={bal_mean:.6g}, max={bal_max:.6g} MW."
        )
    raw_median = float(np.median(raw_balance))
    if abs(raw_median - float(gs_bus.sum())) > 0.05:
        raise ValueError(
            "Uncorrected label residual does not match the network GS total: "
            f"median={raw_median:.6g}, GS={float(gs_bus.sum()):.6g} MW."
        )

    cont_max_abs = 0.0
    if d["cont"] is not None:
        valid = np.where(d["cont"] >= 0)[0]
        if valid.size:
            cont_max_abs = float(
                np.max(np.abs(d["pg"][valid, d["cont"][valid].astype(np.int64)]))
            )
            if cont_max_abs > 0.01:
                raise ValueError(
                    f"Outaged-generator labels are not zero; max_abs={cont_max_abs:.6g} MW."
                )

    flow = audit_flow_labels(op, d, args.device, args.batch_size)
    if flow["label_viol_worst_mean_MW"] > 0.01 or flow["label_viol_max_MW"] > 0.05:
        raise ValueError(
            "Corrected labels fail flow audit: "
            f"mean-worst={flow['label_viol_worst_mean_MW']:.6g}, "
            f"max={flow['label_viol_max_MW']:.6g} MW."
        )

    flow_offset = np.asarray(static["flow_offset"], dtype=np.float32)
    audit = {
        "protocol": "fair4net_l1_v2_phase_aware",
        "case": args.case,
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
        "dimensions": {"nbus": dims[0], "ngen": dims[1], "nbranch": dims[2]},
        "samples": int(d["pd_bus"].shape[0]),
        "split_counts": {"train": 7000, "val": 1000, "test": 2000},
        "demand_contract": "effective_demand=pd_bus+gs_bus",
        "flow_model_contract": trainer.FLOW_MODEL_CONTRACT,
        "pd_bus_includes_gs": False,
        "gs_nonzero_buses": int(np.count_nonzero(np.abs(gs_bus) > 1e-12)),
        "gs_total_MW": float(gs_bus.sum()),
        "gs_sha256": sha256_array(gs_bus),
        "flow_offset_sha256": sha256_array(flow_offset),
        "flow_offset_abs_max_MW": float(np.max(np.abs(flow_offset))) if flow_offset.size else 0.0,
        "phase_shift_nonzero_branches": int(static["phase_shift_nonzero_count"]),
        "fmax_sha256": sha256_array(np.asarray(static["fmax"])),
        "bus_of_gen_sha256": sha256_array(np.asarray(static["bus_of_gen"])),
        "pmin_sha256": sha256_array(np.asarray(static["pmin"])),
        "pmax_sha256": sha256_array(np.asarray(static["pmax"])),
        "corrected_label_balance_abs_mean_MW": bal_mean,
        "corrected_label_balance_abs_max_MW": bal_max,
        "uncorrected_label_balance_median_MW": raw_median,
        "outaged_generator_label_max_abs_MW": cont_max_abs,
        **flow,
        "line_limit_scale": float(trainer.LINE_LIMIT_SCALE),
        "line_limit_convention": (
            "legacy benchmark convention retained; line ampacity and "
            "loading-percent both scaled"
        ),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pandapower": pandapower.__version__,
        },
    }
    json_atomic(args.audit_out, audit)
    print(json.dumps(audit, indent=2))
    print(f"[dataset] audit PASS: {args.audit_out}")


if __name__ == "__main__":
    main()
