#!/usr/bin/env python3
"""Validate mixed clean/line/generator smoke or production artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CASES = {"case300", "case1354pegase", "case2869pegase", "case9241pegase"}
BASELINES = ("mlp", "deepopf", "dc3", "deeplde", "pdl", "reserve", "gnn")
BASELINE_EPOCHS = {
    "mlp": 500,
    "deepopf": 500,
    "dc3": 500,
    "deeplde": 500,
    "pdl": 800,
    "reserve": 500,
    "gnn": 200,
}
EXPECTED_BATCH = {
    "case300": 16,
    "case1354pegase": 8,
    "case2869pegase": 8,
    "case9241pegase": 4,
}


def load_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(label: str, value: Any, *, nonnegative: bool = False) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{label} is non-finite: {value!r}")
    if nonnegative and x < 0.0:
        raise ValueError(f"{label} is negative: {x}")
    return x


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_counts(n: int) -> dict[str, int]:
    train = int(round(0.70 * n))
    val = int(round(0.10 * n))
    return {"0": train, "1": val, "2": n - train - val}


def layout(root: Path, case: str, n: int) -> dict[str, Path]:
    case_root = root / case
    dataset_dir = case_root / "dataset"
    return {
        "case": case_root,
        "dataset_dir": dataset_dir,
        "dataset": dataset_dir / f"dataset_mixed_N{n}_seed0.npz",
        "meta": dataset_dir / f"dataset_mixed_N{n}_seed0.meta.json",
        "audit": dataset_dir / "dataset_audit.json",
        "deq": case_root / "deq_l1_scalar_seed0",
        "baselines": case_root / "baselines_seed0",
    }


def validate_dataset(root: Path, case: str, n: int, candidates: int) -> None:
    p = layout(root, case, n)
    if not (p["dataset_dir"] / ".done").is_file():
        raise FileNotFoundError(p["dataset_dir"] / ".done")
    for key in ("dataset", "meta", "audit"):
        if not p[key].is_file() or p[key].stat().st_size == 0:
            raise FileNotFoundError(p[key])
    meta = load_json(p["meta"])
    audit = load_json(p["audit"])
    if meta.get("case_name") != case or int(meta.get("N", -1)) != n:
        raise ValueError("dataset metadata identity mismatch")
    if meta.get("dataset_schema_name") != "fair4net_mixed_n1_contingency":
        raise ValueError("dataset metadata schema mismatch")
    if meta.get("scenario_contract") != "exactly one of clean, single-line outage, single-generator outage; never simultaneous":
        raise ValueError("dataset scenario contract mismatch")
    if audit.get("status") != "PASS" or audit.get("case") != case or int(audit.get("N", -1)) != n:
        raise ValueError("dataset audit identity/status mismatch")
    if audit.get("split_counts") != split_counts(n):
        raise ValueError(f"unexpected split counts: {audit.get('split_counts')}")
    if int(audit.get("line_candidates", -1)) != candidates:
        raise ValueError("line candidate count mismatch")
    if int(audit.get("generator_candidates", -1)) != candidates:
        raise ValueError("generator candidate count mismatch")
    if int(audit.get("simultaneous_line_generator_samples", -1)) != 0:
        raise ValueError("dataset contains simultaneous line+generator outages")
    if int(audit.get("cross_split_exact_demand_duplicates", -1)) != 0:
        raise ValueError("dataset reuses an exact load scenario across splits")
    isolation = audit.get("split_isolation", {})
    if any(int(isolation.get(key, -1)) != 0 for key in (
        "line_outage_id_overlap", "generator_outage_id_overlap", "exact_demand_row_overlap"
    )):
        raise ValueError("dataset split-isolation attestation failed")
    for name, counts in audit.get("category_counts", {}).items():
        if set(counts) != {"clean", "line", "generator"} or any(int(v) <= 0 for v in counts.values()):
            raise ValueError(f"split={name} lacks one or more scenario categories")
    if set(audit.get("category_counts", {})) != {"train", "val", "test"}:
        raise ValueError("category counts do not cover all three splits")
    for family in ("line", "generator"):
        sets = [
            set(map(int, audit.get(f"{name}_{family}_outage_ids", [])))
            for name in ("train", "validation", "test")
        ]
        if any(not values for values in sets):
            raise ValueError(f"every split must contain {family} outage identities")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError(f"{family} outage identities leak across splits")
        if len(set().union(*sets)) != candidates:
            raise ValueError(f"{family} candidate coverage differs from {candidates}")
    if not bool(audit.get("flow_labels_saved")):
        raise ValueError("independent flow labels are missing")
    if finite("ptdf_change_abs_min", audit.get("ptdf_change_abs_min", math.nan)) <= 1e-8:
        raise ValueError("line outages do not materially change the PTDF")
    if finite("outaged_ptdf_row_abs_max", audit.get("outaged_ptdf_row_abs_max", math.nan), nonnegative=True) > 1e-6:
        raise ValueError("outaged PTDF row is not zero")
    if finite("flow_reconstruction_abs_max_MW", audit.get("flow_reconstruction_abs_max_MW", math.nan), nonnegative=True) > 1e-2:
        raise ValueError("independent PTDF+LODF reconstruction exceeds 0.01 MW")
    if finite("operator_flow_reconstruction_scaled_max", audit.get("operator_flow_reconstruction_scaled_max", math.nan), nonnegative=True) > 1.0:
        raise ValueError("production float32 operator exceeds its atol+rtol bound")
    if finite("balance_abs_max_MW", audit.get("balance_abs_max_MW", math.nan), nonnegative=True) > 1e-2:
        raise ValueError("label balance error exceeds 0.01 MW")
    if finite("outaged_generator_label_abs_max_MW", audit.get("outaged_generator_label_abs_max_MW", math.nan), nonnegative=True) > 1e-3:
        raise ValueError("outaged generator labels are not zero")
    if finite("outaged_line_flow_label_abs_max_MW", audit.get("outaged_line_flow_label_abs_max_MW", math.nan), nonnegative=True) > 1e-3:
        raise ValueError("outaged line-flow labels are not zero")
    sampling = audit.get("opf_sampling", {})
    accepted = int(sampling.get("accepted_total", -1))
    attempts = int(sampling.get("attempts_total", -1))
    rejected = int(sampling.get("rejections_total", -1))
    if sampling.get("policy") != "reject_and_resample_OPFNotConverged_only":
        raise ValueError("OPF rejection policy mismatch")
    if accepted != n or attempts != accepted + rejected:
        raise ValueError("OPF attempt accounting is inconsistent")
    if finite("rejection_rate", sampling.get("rejection_rate", math.nan), nonnegative=True) >= 0.5:
        raise ValueError("OPF rejection rate is at least 50 percent")


def validate_deq(root: Path, case: str, mode: str, n: int, candidates: int) -> None:
    validate_dataset(root, case, n, candidates)
    out = layout(root, case, n)["deq"]
    if not (out / ".done").is_file():
        raise FileNotFoundError(out / ".done")
    stem = f"deq_{case}_mixed_smoke" if mode == "smoke" else f"deq_{case}_mixed_l1_scalar_e30"
    history = load_json(out / f"{stem}_latest.json")
    expected_epochs = 1 if mode == "smoke" else 30
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError(f"DEQ history must contain {expected_epochs} epochs")
    if int(history[-1].get("epoch", -1)) != expected_epochs:
        raise ValueError("DEQ final epoch identity mismatch")
    row = history[-1]
    rho = finite("rho", row.get("rho", math.nan))
    stability = finite("tau_sigma_K2", row.get("tau_sigma_K2", math.nan), nonnegative=True)
    if not 0.0 < rho < 1.0 or stability >= 1.0:
        raise ValueError("DEQ learned step sizes violate the PDHG stability contract")
    for split in ("train", "val"):
        metrics = row.get(split, {})
        for key in (
            "L_total", "J_cost", "J_bal", "J_flow", "J_flow_l1", "fwd_res", "alg_res",
            "bal_abs_MW", "viol_MW", "viol_p95_MW", "viol_max_MW", "rmse_MW_diag",
            "cost_rel_gap_abs", "bal_sat_rate", "flow_sat_rate",
        ):
            finite(f"{split}.{key}", metrics.get(key, math.nan), nonnegative=(key != "J_cost"))
        if float(metrics["fwd_res"]) >= 1.0 or float(metrics["alg_res"]) >= 1.0:
            raise ValueError(f"{split} fixed-point residual is at least one")
    for suffix in (".pt", "_training_state.pt"):
        path = out / f"{stem}{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if mode == "full":
        summary = load_json(out / f"{stem}_summary.json")
        evaluation = load_json(out / f"{stem}_eval_vs_gt_summary.json")
        config = summary.get("config", {})
        expected = {
            "epochs": 30,
            "batch_size": EXPECTED_BATCH[case],
            "lr": 1e-4,
            "seed_split": 0,
            "step_mode": "scalar",
            "picard_iters": 50,
            "anderson_m": 4,
            "anderson_lam": 1e-3,
            "anderson_max_iter": 600,
            "anderson_tol": 1e-10,
            "b_anderson_m": 10,
            "b_anderson_lam": 1e-4,
            "b_anderson_max_iter": 100,
            "b_anderson_tol": 1e-6,
            "w_cost": 1.0,
            "w_bal": 1e5,
            "w_flow": 1e8,
            "w_flow_l1": 1e4,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"DEQ config mismatch for {key}: {config.get(key)!r} != {value!r}")
        if summary.get("case_name") != case:
            raise ValueError("DEQ case mismatch")
        if not summary.get("physics", {}).get("lodf_sha256"):
            raise ValueError("DEQ summary lacks the LODF digest")
        test = summary.get("test", {})
        for key in (
            "bal_abs_MW", "bal_max_MW", "viol_MW", "viol_p95_MW", "viol_max_MW",
            "rmse_MW_diag", "cost_rel_gap_abs", "fwd_res", "alg_res",
        ):
            finite(f"DEQ.test.{key}", test.get(key, math.nan), nonnegative=True)
        if float(test["fwd_res"]) >= 1e-2 or float(test["alg_res"]) >= 1e-2:
            raise ValueError("DEQ final test fixed-point residual is at least 1e-2")
        if int(evaluation.get("N", -1)) != split_counts(n)["2"]:
            raise ValueError("DEQ evaluation summary has the wrong test size")
        categories = evaluation.get("category_summary", {})
        if set(categories) != {"clean", "line", "generator"}:
            raise ValueError("DEQ evaluation lacks clean/line/generator category summaries")
        if sum(int(row.get("N", 0)) for row in categories.values()) != split_counts(n)["2"]:
            raise ValueError("DEQ category-summary sample counts do not equal the test size")
        eval_csv = out / f"{stem}_eval_vs_gt.csv"
        if not eval_csv.is_file() or eval_csv.stat().st_size == 0:
            raise FileNotFoundError(eval_csv)


def validate_baselines(root: Path, case: str, mode: str, n: int, candidates: int) -> None:
    validate_dataset(root, case, n, candidates)
    p = layout(root, case, n)
    out = p["baselines"]
    if not (out / ".done").is_file():
        raise FileNotFoundError(out / ".done")
    expected_epochs = {method: 1 for method in BASELINES} if mode == "smoke" else BASELINE_EPOCHS
    rows = []
    expected_splits = split_counts(n)
    expected_hash = file_sha256(p["dataset"])
    for method in BASELINES:
        row = load_json(out / f"{method}.json")
        if row.get("case") != case or row.get("baseline") != method:
            raise ValueError(f"baseline identity mismatch: {method}")
        if int(row.get("N", -1)) != n or row.get("split") != "mixed_disjoint":
            raise ValueError(f"baseline dataset/split mismatch: {method}")
        if row.get("contingency_encoding") != "lodf_column+pmax_zero":
            raise ValueError(f"baseline contingency encoding mismatch: {method}")
        if row.get("scenario_protocol") != "fair4net_mixed_n1_contingency_v1":
            raise ValueError(f"baseline scenario protocol mismatch: {method}")
        if row.get("dataset_sha256") != expected_hash:
            raise ValueError(f"baseline dataset digest mismatch: {method}")
        if int(row.get("epochs", -1)) != expected_epochs[method]:
            raise ValueError(f"baseline epoch mismatch: {method}")
        counts = (
            int(row.get("train_samples", -1)),
            int(row.get("val_samples", -1)),
            int(row.get("test_samples", -1)),
        )
        if counts != (expected_splits["0"], expected_splits["1"], expected_splits["2"]):
            raise ValueError(f"baseline split counts mismatch: {method}")
        for key in (
            "bal_abs_MW", "bal_max_MW", "viol_MW", "viol_p95_MW",
            "viol_branch_mean_MW", "viol_max_MW", "J_flow", "J_flow_l1",
            "rmse_MW", "cost_rel_gap_abs", "train_time_sec",
        ):
            finite(f"{method}.{key}", row.get(key, math.nan), nonnegative=True)
        if mode == "full":
            category_total = 0
            for category in ("clean", "line", "generator"):
                count = int(row.get(f"{category}_samples", -1))
                if count <= 0:
                    raise ValueError(f"{method} has no {category} test samples")
                category_total += count
                for key in (
                    "bal_abs_MW", "bal_max_MW", "viol_MW", "viol_p95_MW",
                    "viol_max_MW", "flow_sat_rate", "rmse_MW", "cost_rel_gap_abs",
                ):
                    finite(
                        f"{method}.{category}_{key}",
                        row.get(f"{category}_{key}", math.nan),
                        nonnegative=True,
                    )
            if category_total != expected_splits["2"]:
                raise ValueError(f"{method} category counts do not equal the test size")
        rows.append(row)
    if len(load_json(out / "baseline_summary.json").get("rows", [])) != len(BASELINES):
        raise ValueError("baseline summary does not contain all seven methods")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--mode", required=True, choices=("smoke", "full"))
    ap.add_argument("--role", default="all", choices=("dataset", "deq", "baselines", "all"))
    ap.add_argument("--expected-N", type=int)
    args = ap.parse_args()
    n = int(args.expected_N if args.expected_N is not None else (120 if args.mode == "smoke" else 10000))
    candidates = 8 if args.mode == "smoke" else 32
    if args.role in {"dataset", "all"}:
        validate_dataset(args.root, args.case, n, candidates)
    if args.role in {"deq", "all"}:
        validate_deq(args.root, args.case, args.mode, n, candidates)
    if args.role in {"baselines", "all"}:
        validate_baselines(args.root, args.case, args.mode, n, candidates)
    print(f"[validate-mixed-multinet] PASS case={args.case} mode={args.mode} role={args.role}")


if __name__ == "__main__":
    main()
