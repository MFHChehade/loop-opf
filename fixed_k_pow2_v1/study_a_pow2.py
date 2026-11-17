#!/usr/bin/env python3
"""Matched clean-OPF fixed-K sweep with one rollout per arm and test batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


GRID = tuple(2 ** k for k in range(14))
CASES = ("case300", "case1354pegase", "case2869pegase", "case9241pegase")
NINE_K_HASHES = {
    "dataset_sha256": "78291488245b4d9024ea0ebd06e9211a8ad7c5966a6b05835e96c493cdfa875f",
    "audit_sha256": "861a056fc7177daad7a8607f1eebfd30951ffbfc721eeb21ab4b733286f5cff1",
    "checkpoint_sha256": "07ddd63046152b95a2f377fe21128caf445eeefa890945cea129d8117cf48675",
    "source_summary_sha256": "ea1232d99704bd06a19b84a8b3a59cd087b49dfd85eed53130b4dd3889f5b28d",
}
NINE_K_AUDITED_SIDECAR_SHA256 = "3fcd6f8f7405741e65274df802de1f7fd83f19e84a95fe34f0f30d663549b401"


def grid(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or values != sorted(set(values)) or any(x & (x - 1) or x < 1 for x in values):
        raise ValueError("K must be distinct, increasing powers of two")
    return values


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assert_reference(rows: list[dict], reference: Path | None) -> None:
    if reference is None:
        return
    if not reference.is_file():
        raise FileNotFoundError(reference)
    with reference.open(newline="") as stream:
        old = {int(r["K"]): r for r in csv.DictReader(stream)}
    for row in rows:
        k = int(row["K"])
        if k not in old:
            continue
        for field in ("rmse_MW", "bal_abs_MW", "viol_worst_mean_MW"):
            target = float(old[k][field])
            actual = float(row[field])
            limit = max(1e-4, 1e-4 * abs(target))
            if not np.isfinite(actual) or abs(actual - target) > limit:
                raise RuntimeError(
                    f"parity failure K={k} {field}: new={actual} old={target} limit={limit}"
                )
    print(f"[study-a] reference parity PASS {reference}", flush=True)


@torch.inference_mode()
def sweep(op, data: dict, test: np.ndarray, ks: list[int], batch_size: int, harness, device):
    wanted = set(ks)
    preds = {k: [] for k in ks}
    residuals = {k: [] for k in ks}
    elapsed = {k: 0.0 for k in ks}
    demands, targets = [], []
    dimension = int(op.ngen) + 1 + 2 * int(op.nbranch)
    for offset in range(0, len(test), batch_size):
        ids = test[offset:offset + batch_size]
        d = torch.as_tensor(data["pd_bus"][ids], dtype=torch.float32, device=device)
        target = torch.as_tensor(data["pg"][ids], dtype=torch.float32, device=device)
        cont = None if data["cont"] is None else torch.as_tensor(data["cont"][ids], dtype=torch.long, device=device)
        z = torch.zeros((len(ids), dimension), dtype=d.dtype, device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for step in range(1, ks[-1] + 1):
            z = op(z, d, cont)
            if step in wanted:
                next_z = op(z, d, cont)
                residuals[step].append(
                    ((next_z - z).norm(dim=1) / (next_z.norm(dim=1) + 1e-12)).clone()
                )
                preds[step].append(z[:, :op.ngen].clone())
                torch.cuda.synchronize(device)
                elapsed[step] += time.perf_counter() - started
        demands.append(d)
        targets.append(target)
        if (offset // batch_size + 1) % 25 == 0:
            print(f"[study-a] batches={offset // batch_size + 1}/{(len(test) + batch_size - 1) // batch_size}", flush=True)
    d_all = torch.cat(demands)
    target_all = torch.cat(targets)
    rows = []
    for k in ks:
        metrics = harness.score(op, torch.cat(preds[k]), d_all, target_all)
        rows.append({
            "K": k, "operator_evaluations_per_sample": k, "test_samples": len(test),
            "rmse_MW": float(metrics["rmse_MW"]),
            "bal_abs_MW": float(metrics["bal_abs_MW"]),
            "viol_worst_mean_MW": float(metrics["viol_worst_mean_MW"]),
            "cost_gap_abs_percent": 100.0 * float(metrics["cost_rel_gap_abs"]),
            "bal_sat_rate": float(metrics["bal_sat_rate"]),
            "flow_sat_rate": float(metrics["flow_sat_rate"]),
            "fixed_point_residual_mean": float(torch.cat(residuals[k]).mean()),
            "wall_seconds": elapsed[k],
        })
        print("[study-a]", json.dumps(rows[-1], sort_keys=True), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--compare-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--source-stem", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path)
    ap.add_argument("--reference-initial", type=Path)
    ap.add_argument("--reference-learned", type=Path)
    ap.add_argument("--k-values", type=grid, default=list(GRID))
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--expected-test", type=int, default=2000)
    args = ap.parse_args()
    if args.batch_size != 8:
        raise ValueError("the matched protocol requires batch size 8")
    if os.environ.get("DEQ_CASE") != args.case:
        raise ValueError("DEQ_CASE must select the requested network before import")
    sys.path.insert(0, str(args.study_dir.resolve()))
    sys.path.insert(0, str(args.compare_dir.resolve()))
    import baseline_harness_fair4net as harness
    import case300_fixed_k_inference_budget as base
    import compare_steps
    import deq_trainer_fair4net as trainer

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("this study requires a Vista GPU node")
    np.random.seed(0)
    torch.manual_seed(0)
    checkpoint_path = Path(str(args.source_stem) + ".pt")
    summary_path = Path(str(args.source_stem) + "_summary.json")
    audit, summary, hashes = compare_steps.validate_generic_source(
        base, args.data, args.audit, checkpoint_path, summary_path, args.case
    )
    expected = NINE_K_HASHES if args.case == "case9241pegase" else base.EXPECTED_ARTIFACTS[args.case]
    for key, value in expected.items():
        if hashes[key] != value:
            raise ValueError(f"{key} differs from the immutable source artifact")
    # The 9k byte hash changes across GPU nodes despite numerical agreement
    # with its audited sidecar. The checks below replace only that hash gate.
    learned, data, checkpoint, flow_provenance = base.build_operator_and_data(
        args.data, audit, summary, checkpoint_path, args.case, device,
        args.case == "case9241pegase",
    )
    if args.case == "case9241pegase":
        if data.get("cont_line") is not None and np.any(np.asarray(data["cont_line"]) >= 0):
            raise ValueError("line-outage data cannot be used with the clean 9k checkpoint")
        if args.sidecar is None or not args.sidecar.is_file():
            raise FileNotFoundError("9k audited flow-offset sidecar required")
        if sha256(args.sidecar) != NINE_K_AUDITED_SIDECAR_SHA256:
            raise ValueError("9k audited flow-offset sidecar hash mismatch")
        sidecar = np.load(args.sidecar)
        actual = learned.flow_offset.detach().cpu().numpy()
        error = float(np.max(np.abs(actual - sidecar)))
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f"9k flow-offset sidecar mismatch: {error} MW")
        flow_provenance["audited_sidecar_max_abs_error_MW"] = error
        flow_provenance["audited_sidecar_tolerance_MW"] = 1e-5
        flow_provenance["audited_sidecar_sha256"] = sha256(args.sidecar)
    initial = compare_steps.make_initial_step_operator(learned, trainer, checkpoint, 0.9, 0.0, device)
    test = np.flatnonzero(np.asarray(data["split"], dtype=np.int64) == 2)
    if args.max_test_samples:
        test = test[:args.max_test_samples]
    elif len(test) != args.expected_test:
        raise ValueError(f"expected {args.expected_test} test rows, got {len(test)}")
    if not len(test):
        raise ValueError("empty test set")
    arms = (
        ("initial", initial, args.reference_initial),
        ("learned", learned, args.reference_learned),
    )
    for name, op, reference in arms:
        out = args.out_root / args.case / name
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing output: {out}")
        rows = sweep(op, data, test, args.k_values, args.batch_size, harness, device)
        assert_reference(rows, reference)
        write_rows(out / "quality_by_k.csv", rows)
        write_json(out / "result.json", {
            "status": "PASS", "study": "clean_fixed_k_initial_vs_learned_pow2",
            "case": args.case, "arm": name, "K": args.k_values,
            "test_samples": len(test), "initial_state": "z0=0",
            "anderson": False, "residual_stopping": False, "dtype": "float32",
            "batch_size": args.batch_size, "checkpoint_immutable": True,
            "source_hashes": hashes, "flow_offset_provenance": flow_provenance,
            "steps": compare_steps.steps_record(op.steps, torch.float32, device),
            "reference": str(reference) if reference else None,
            "rows": rows,
        })


if __name__ == "__main__":
    main()
